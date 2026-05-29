#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
# src/04_portfolio/monte_carlo.py

"""
Simulation de Monte Carlo par copule skewed-t (Demarta & McNeil 2005)
pour l'optimisation moyenne-CVaR.

Suit fidèlement les Appendices B, C et D du papier :
    Zhao, Stasinakis, Sermpinis & Da Silva Fernandes (2019)
    Int J Fin Econ, 24, 1443-1463.

Structure :
    1. Distribution skewed-t de Hansen (1994) — modèles marginaux (Appendice B)
       - PDF, CDF, PPF (inverse CDF), estimation MLE
    2. Copule GH skewed-t (Demarta & McNeil 2005) — dépendance multivariée (Appendice D)
       - CDF marginale via quadrature de Gauss-Laguerre généralisée
       - Simulation par mélange normal-inverse gamma
    3. Estimation des paramètres de la copule sur fenêtre roulante
    4. Pipeline complet de simulation des rendements de portefeuille
"""

import warnings
import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_func, roots_genlaguerre
from scipy.stats import (
    t as scipy_t,
    norm as scipy_norm,
    invgamma as scipy_invgamma,
)
from scipy.optimize import minimize, brentq

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ──────────────────────────────────────────────────────────────────────────────

FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]
N_FACTORS = 5

# Noeuds Gauss-Laguerre généralisés (pré-calculés, partagés)
_GL_NODES_CACHE: dict = {}

# Paramètres de la grille pour la CDF marginale GH
_GRID_X_MIN = -15.0
_GRID_X_MAX = 15.0
_GRID_N     = 600       # nombre de points sur la grille


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DISTRIBUTION SKEWED-T DE HANSEN (1994) — MARGINALES
# ══════════════════════════════════════════════════════════════════════════════

def _hansen_constants(nu: float, lam: float) -> tuple[float, float, float]:
    """
    Calcule les constantes a, b, c de la distribution skewed-t de Hansen (1994).

    Paramètres
    ----------
    nu  : degrés de liberté (> 2)
    lam : asymétrie (-1 < lam < 1)

    Retourne
    --------
    (a, b, c) avec :
        c = Γ((nu+1)/2) / (√(π(nu-2)) Γ(nu/2))
        a = 4λc(nu-2)/(nu-1)
        b = √(1 + 3λ² - a²)
    """
    c = gamma_func((nu + 1.0) / 2.0) / (
        np.sqrt(np.pi * (nu - 2.0)) * gamma_func(nu / 2.0)
    )
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam**2 - a**2)
    return a, b, c


def hansen_skt_pdf(z: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """
    PDF de la distribution skewed-t de Hansen (1994).
    Hansen (1994) Int'l Economic Review, eq. (5).

    Paramètres
    ----------
    z   : valeurs où évaluer la PDF (array ou scalaire)
    nu  : degrés de liberté (> 2)
    lam : asymétrie (-1 < lam < 1)

    Convention de normalisation : variance unitaire.
    """
    a, b, c = _hansen_constants(nu, lam)
    z = np.asarray(z, dtype=float)
    bz_plus_a = b * z + a
    factor = 1.0 / (nu - 2.0)

    left  = b * c * (1.0 + factor * (bz_plus_a / (1.0 - lam)) ** 2) ** (-(nu + 1.0) / 2.0)
    right = b * c * (1.0 + factor * (bz_plus_a / (1.0 + lam)) ** 2) ** (-(nu + 1.0) / 2.0)
    return np.where(bz_plus_a < 0.0, left, right)


def hansen_skt_cdf(z: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """
    CDF de la distribution skewed-t de Hansen (1994).

    La distribution t utilisée dans Hansen (1994) a pour kernel
    (1 + x²/(nu-2))^{-(nu+1)/2}, soit la t de scipy(df=nu)
    redimensionnée par √((nu-2)/nu).

    Facteur de conversion : F_Hansen(x; nu) = F_scipy(x · √(nu/(nu-2)); df=nu)
    """
    a, b, c = _hansen_constants(nu, lam)
    z = np.asarray(z, dtype=float)
    scale = np.sqrt(nu / (nu - 2.0))       # facteur de conversion Hansen → scipy
    bz_plus_a = b * z + a

    cdf_left  = (1.0 - lam) * scipy_t.cdf(
        bz_plus_a / (1.0 - lam) * scale, df=nu
    )
    cdf_right = (
        (1.0 - lam) / 2.0
        + (1.0 + lam) * (scipy_t.cdf(bz_plus_a / (1.0 + lam) * scale, df=nu) - 0.5)
    )
    return np.where(bz_plus_a < 0.0, cdf_left, cdf_right)


def hansen_skt_ppf(p: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """
    Fonction quantile (inverse CDF) de la distribution skewed-t de Hansen (1994).

    Pour p < (1-λ)/2 :
        z = [(1-λ) · F_scipy^{-1}(p/(1-λ); df=nu) / scale - a] / b
    Pour p ≥ (1-λ)/2 :
        z = [(1+λ) · F_scipy^{-1}((p+λ)/(1+λ); df=nu) / scale - a] / b
    """
    a, b, c = _hansen_constants(nu, lam)
    p = np.asarray(p, dtype=float)
    scale = np.sqrt(nu / (nu - 2.0))
    inv_scale = 1.0 / scale
    threshold = (1.0 - lam) / 2.0

    # Branche gauche
    t_q_left  = scipy_t.ppf(np.clip(p / (1.0 - lam), 1e-10, 1.0 - 1e-10), df=nu)
    z_left    = ((1.0 - lam) * t_q_left * inv_scale - a) / b

    # Branche droite
    t_q_right = scipy_t.ppf(np.clip((p + lam) / (1.0 + lam), 1e-10, 1.0 - 1e-10), df=nu)
    z_right   = ((1.0 + lam) * t_q_right * inv_scale - a) / b

    return np.where(p < threshold, z_left, z_right)


def fit_hansen_skt(residuals: np.ndarray) -> tuple[float, float]:
    """
    Estimation MLE des paramètres (nu, lam) de la distribution skewed-t
    de Hansen (1994) sur une série de résidus standardisés.

    Paramètres
    ----------
    residuals : résidus standardisés (1D array)

    Retourne
    --------
    (nu_hat, lam_hat) : degrés de liberté et asymétrie estimés
    """
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 20:
        return 8.0, 0.0    # valeurs par défaut si pas assez d'observations

    def neg_loglik(params):
        nu_, lam_ = params
        if nu_ <= 2.01 or abs(lam_) >= 0.999:
            return 1e10
        pdf_vals = hansen_skt_pdf(residuals, nu_, lam_)
        pdf_vals = np.where(pdf_vals > 1e-300, pdf_vals, 1e-300)
        return -np.sum(np.log(pdf_vals))

    # Points de départ multiples
    best_val = np.inf
    best_x   = [8.0, 0.0]
    for nu0 in [5.0, 8.0, 15.0]:
        for lam0 in [-0.1, 0.0, 0.1]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = minimize(
                        neg_loglik,
                        x0=[nu0, lam0],
                        method="Nelder-Mead",
                        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 1000},
                    )
                if res.fun < best_val and res.x[0] > 2.01 and abs(res.x[1]) < 0.999:
                    best_val = res.fun
                    best_x   = res.x.tolist()
            except Exception:
                continue

    nu_hat  = float(np.clip(best_x[0], 2.5, 50.0))
    lam_hat = float(np.clip(best_x[1], -0.95, 0.95))
    return nu_hat, lam_hat


# ══════════════════════════════════════════════════════════════════════════════
# 2.  COPULE GH SKEWED-T (Demarta & McNeil 2005) — DÉPENDANCE MULTIVARIÉE
# ══════════════════════════════════════════════════════════════════════════════

def _get_gl_nodes(alpha: float, n_nodes: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Noeuds et poids de la quadrature de Gauss-Laguerre généralisée
    ∫₀^∞ f(u) u^alpha e^{-u} du ≈ Σ_k w_k f(u_k).

    Mise en cache par (alpha, n_nodes).
    """
    key = (round(alpha, 6), n_nodes)
    if key not in _GL_NODES_CACHE:
        nodes, weights = roots_genlaguerre(n_nodes, alpha)
        _GL_NODES_CACHE[key] = (nodes, weights)
    return _GL_NODES_CACHE[key]


def gh_skt_marginal_cdf_grid(
    nu_c: float,
    gamma_ci: float,
    n_nodes: int = 40,
    x_min: float = _GRID_X_MIN,
    x_max: float = _GRID_X_MAX,
    n_grid: int = _GRID_N,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pré-calcule la CDF marginale de la copule GH skewed-t sur une grille régulière.

    La variable marginale X_i suit :
        X_i | W ~ N(γ_i W, W)    où    W ~ InvGamma(ν_c/2, ν_c/2)

    Par substitution U = ν_c/(2W), U ~ Gamma(ν_c/2, 1), on obtient :

        F(x) = E_U[ Φ(x√(2U/ν_c) − γ_i√(ν_c/(2U))) ]
             = (1/Γ(a)) ∫₀^∞ Φ(x√(2u/ν_c) − γ_i√(ν_c/(2u))) · u^{a-1} e^{-u} du

    avec a = ν_c/2. Calculé via quadrature Gauss-Laguerre généralisée (ordre α=a-1).

    Paramètres
    ----------
    nu_c     : degrés de liberté de la copule (> 2)
    gamma_ci : asymétrie copule pour le facteur i
    n_nodes  : nombre de noeuds de quadrature (défaut 40)
    x_min, x_max, n_grid : paramètres de la grille

    Retourne
    --------
    (x_grid, cdf_grid) : grille et valeurs de CDF correspondantes
    """
    a      = nu_c / 2.0
    alpha  = a - 1.0           # paramètre GL généralisé
    nodes, weights = _get_gl_nodes(alpha, n_nodes)

    # Termes de la quadrature vectorisés sur la grille x et les noeuds u
    x_grid   = np.linspace(x_min, x_max, n_grid)
    # x_grid : (n_grid,), nodes : (n_nodes,)
    # arg = x * sqrt(2u/nu_c) - gamma * sqrt(nu_c / (2u))
    sqrt_2u_over_nu = np.sqrt(2.0 * nodes / nu_c)    # (n_nodes,)
    sqrt_nu_over_2u = np.sqrt(nu_c / (2.0 * nodes))  # (n_nodes,)

    # Broadcasting : (n_grid, n_nodes)
    arg = (
        x_grid[:, np.newaxis] * sqrt_2u_over_nu[np.newaxis, :]
        - gamma_ci * sqrt_nu_over_2u[np.newaxis, :]
    )
    phi_arg = scipy_norm.cdf(arg)   # (n_grid, n_nodes)

    # Intégrale = (1/Γ(a)) * Σ_k w_k Φ(arg_k)
    cdf_grid = phi_arg @ weights / gamma_func(a)

    # Clip pour s'assurer que la CDF est dans [eps, 1-eps]
    cdf_grid = np.clip(cdf_grid, 1e-10, 1.0 - 1e-10)
    return x_grid, cdf_grid


def _cdf_from_grid(
    x_vals: np.ndarray,
    x_grid: np.ndarray,
    cdf_grid: np.ndarray,
) -> np.ndarray:
    """Évalue la CDF pré-calculée par interpolation linéaire."""
    return np.interp(x_vals, x_grid, cdf_grid,
                     left=1e-10, right=1.0 - 1e-10)


def _ppf_from_grid(
    u_vals: np.ndarray,
    x_grid: np.ndarray,
    cdf_grid: np.ndarray,
) -> np.ndarray:
    """Évalue la fonction quantile par interpolation linéaire (grille inversée)."""
    return np.interp(u_vals, cdf_grid, x_grid,
                     left=x_grid[0], right=x_grid[-1])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ESTIMATION DES PARAMÈTRES DE LA COPULE
# ══════════════════════════════════════════════════════════════════════════════

def _tcop_loglik(
    nu_c: float,
    u_window: np.ndarray,
    R_window: list[np.ndarray],
) -> float:
    """
    Log-vraisemblance de la copule t symétrique (cas particulier γ_c = 0).

    Utilisée pour estimer ν_c par MLE (étape 1 de la procédure en 2 étapes).

    log L_t = Σ_t [log f_t(x_t; R_t, ν_c) - Σ_i log f_t(x_{it}; ν_c)]

    où x_{it} = F_t^{-1}(u_{it}; df=ν_c) et F_t est la CDF de Student(ν_c).
    """
    if nu_c <= 2.01:
        return -1e10
    T, n = u_window.shape

    # Transform uniforms to t quantiles
    u_clipped = np.clip(u_window, 1e-6, 1.0 - 1e-6)
    x_mat = scipy_t.ppf(u_clipped, df=nu_c)   # (T, n)

    # Univariate t log-densities : log f_t(x_{it}; nu_c)
    log_f_univ = scipy_t.logpdf(x_mat, df=nu_c)  # (T, n)

    # Multivariate t log-density at each date
    log_f_multi = np.zeros(T)
    lnu = nu_c / 2.0
    log_c_mv = (
        gamma_func(lnu + n / 2.0) / gamma_func(lnu)
        / (np.pi * (nu_c - 2.0)) ** (n / 2.0)
    )
    if log_c_mv <= 0:
        return -1e10
    log_const = np.log(log_c_mv)

    for t_idx in range(T):
        R_t = R_window[t_idx]
        x_t = x_mat[t_idx]
        sign, logdet = np.linalg.slogdet(R_t)
        if sign <= 0:
            log_f_multi[t_idx] = -1e10
            continue
        try:
            R_inv = np.linalg.inv(R_t)
        except np.linalg.LinAlgError:
            log_f_multi[t_idx] = -1e10
            continue
        Q_t = float(x_t @ R_inv @ x_t)
        log_f_multi[t_idx] = (
            log_const
            - 0.5 * logdet
            - (lnu + n / 2.0) * np.log(1.0 + Q_t / (nu_c - 2.0))
        )

    ll = np.sum(log_f_multi) - np.sum(log_f_univ)
    return float(ll)


def estimate_copula_params(
    u_window: np.ndarray,
    R_window: list[np.ndarray],
    nu0: float = 8.0,
    gamma0: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """
    Estimation des paramètres (ν_c, γ_c) de la copule GH skewed-t sur une fenêtre.

    Procédure en 2 étapes (IFM simplifiée) :
        Étape 1 : MLE de ν_c via la copule t symétrique (γ_c = 0)
        Étape 2 : Estimation de γ_{c,i} par méthode des moments :
                  γ_{c,i} ≈ [(nu_c - 2) / nu_c] * mean(F_t^{-1}(u_i))

    La copule t est le cas particulier γ_c = 0 de la copule GH skewed-t.
    L'asymétrie résiduelle dans les u_{i,t} est capturée par γ_{c,i}.

    Paramètres
    ----------
    u_window : (T, n) — valeurs PIT uniformes sur la fenêtre
    R_window : liste de T matrices de corrélation (n × n)
    nu0      : point de départ pour ν_c
    gamma0   : point de départ pour γ_c (None → zéros)

    Retourne
    --------
    (nu_c, gamma_c) : degrés de liberté et vecteur d'asymétrie copule
    """
    n = u_window.shape[1]
    if gamma0 is None:
        gamma0 = np.zeros(n)

    # Étape 1 : MLE de nu_c via copule t
    def neg_ll(nu_arr):
        return -_tcop_loglik(float(nu_arr[0]), u_window, R_window)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            neg_ll,
            x0=[nu0],
            method="Nelder-Mead",
            options={"xatol": 0.1, "fatol": 0.5, "maxiter": 200},
        )

    nu_c = float(np.clip(res.x[0] if res.success else nu0, 3.0, 50.0))

    # Étape 2 : γ_{c,i} par méthode des moments
    # E[X_i] = γ_{c,i} * E[W] = γ_{c,i} * nu_c/(nu_c-2)
    # → γ_{c,i} ≈ mean(x_{i,t}) * (nu_c-2)/nu_c
    u_clipped = np.clip(u_window, 1e-6, 1.0 - 1e-6)
    x_mat     = scipy_t.ppf(u_clipped, df=nu_c)  # (T, n)
    gamma_c   = x_mat.mean(axis=0) * (nu_c - 2.0) / nu_c
    gamma_c   = np.clip(gamma_c, -2.0, 2.0)

    return nu_c, gamma_c


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SIMULATION DE MONTE CARLO — COPULE GH SKEWED-T
# ══════════════════════════════════════════════════════════════════════════════

def simulate_gh_skt_copula(
    R_t: np.ndarray,
    nu_c: float,
    gamma_c: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simule n_sims réalisations de la copule GH skewed-t (Demarta & McNeil 2005).

    Algorithme (Appendice D, construction variance-mean mixture) :
        1. W_m ~ InvGamma(ν_c/2, ν_c/2)
        2. Z_m ~ N(0, R_t)  [multivariée avec matrice de corrélation R_t]
        3. X_m = γ_c · W_m + √W_m · Z_m
        4. U_{m,i} = F_{X_i}(X_{m,i})  [CDF marginale par grille GL pré-calculée]

    Paramètres
    ----------
    R_t     : matrice de corrélation (n × n) à la date t
    nu_c    : degrés de liberté de la copule
    gamma_c : vecteur d'asymétrie (n,)
    n_sims  : nombre de simulations
    rng     : générateur aléatoire numpy

    Retourne
    --------
    U : (n_sims, n) — échantillons uniformes de la copule
    """
    n = len(gamma_c)

    # Étape 1 : W_m ~ InvGamma(a, a) avec a = nu_c/2
    a = nu_c / 2.0
    # scipy.stats.invgamma : a = alpha (forme), scale = beta (paramètre d'échelle)
    W_m = scipy_invgamma.rvs(a=a, scale=a, size=n_sims, random_state=rng)  # (n_sims,)

    # Étape 2 : Z_m ~ N(0, R_t) via décomposition de Cholesky
    # Correction PSD si nécessaire
    R_psd = _ensure_psd(R_t)
    try:
        L = np.linalg.cholesky(R_psd)
    except np.linalg.LinAlgError:
        # Fallback : décomposition spectrale
        eigvals, eigvecs = np.linalg.eigh(R_psd)
        eigvals = np.maximum(eigvals, 1e-8)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    Z_m = rng.standard_normal((n_sims, n)) @ L.T   # (n_sims, n)

    # Étape 3 : X_m = γ_c · W_m + √W_m · Z_m
    X_m = (
        gamma_c[np.newaxis, :] * W_m[:, np.newaxis]
        + np.sqrt(W_m[:, np.newaxis]) * Z_m
    )   # (n_sims, n)

    # Étape 4 : U_{m,i} = F_{X_i}(X_{m,i}) via grille GL pré-calculée
    U = np.empty((n_sims, n), dtype=float)
    for i in range(n):
        x_grid, cdf_grid = gh_skt_marginal_cdf_grid(nu_c, gamma_c[i])
        U[:, i] = _cdf_from_grid(X_m[:, i], x_grid, cdf_grid)

    return U


def simulate_portfolio_returns(
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    R_t: np.ndarray,
    nu_c: float,
    gamma_c: np.ndarray,
    marginal_params: list[tuple[float, float]],
    n_sims: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simule n_sims vecteurs de rendements mensuels des 5 facteurs à la date t.

    Pipeline complet (Section 5.2 et Appendices B, D) :
        1. U_{m,i} ~ Copule GH skewed-t (R_t, ν_c, γ_c)
        2. Z_{m,i} = G_i^{-1}(U_{m,i} ; η_i, λ_i^H)   [inverse skewed-t de Hansen]
        3. r_{m,i,t} = μ_{i,t} + σ_{i,t} · Z_{m,i}

    Paramètres
    ----------
    mu_t             : prévisions de rendements (n,) à la date t
    sigma_t          : volatilités conditionnelles GARCH (n,) à la date t
    R_t              : matrice de corrélation (n × n)
    nu_c             : degrés de liberté copule
    gamma_c          : asymétrie copule (n,)
    marginal_params  : liste de (η_i, λ_i) pour chaque facteur i
    n_sims           : nombre de simulations Q
    seed             : graine aléatoire (optionnel)

    Retourne
    --------
    r_sim : (n_sims, n) — rendements simulés
    """
    rng = np.random.default_rng(seed)

    # Étape 1 : uniforms de la copule
    U = simulate_gh_skt_copula(R_t, nu_c, gamma_c, n_sims, rng)  # (Q, n)

    # Étape 2 : résidus standardisés via inverse CDF de Hansen skewed-t
    Z = np.empty_like(U)
    for i, (eta_i, lam_i) in enumerate(marginal_params):
        u_i = np.clip(U[:, i], 1e-8, 1.0 - 1e-8)
        Z[:, i] = hansen_skt_ppf(u_i, eta_i, lam_i)

    # Étape 3 : reconstruction des rendements
    # r_{m,i,t} = μ_{i,t} + σ_{i,t} · Z_{m,i}
    r_sim = mu_t[np.newaxis, :] + sigma_t[np.newaxis, :] * Z   # (Q, n)

    return r_sim


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_psd(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Projette une matrice de corrélation sur le cône des matrices semi-définies
    positives (correction spectrale minimale).
    """
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 1.0)
    eigvals = np.linalg.eigvalsh(R)
    if np.any(eigvals < eps):
        R += (eps - eigvals.min()) * np.eye(len(R))
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        np.fill_diagonal(R, 1.0)
    return R


def corr_row_to_matrix(
    row: pd.Series,
    facteurs_corr: list[str],
) -> np.ndarray:
    """
    Reconstruit une matrice de corrélation (5×5) depuis une ligne de paires
    au format 'FACTEUR_A-FACTEUR_B' (triangle supérieur).
    """
    n = len(facteurs_corr)
    R = np.eye(n)
    for i, fi in enumerate(facteurs_corr):
        for j, fj in enumerate(facteurs_corr):
            if j <= i:
                continue
            pair1 = f"{fi}-{fj}"
            pair2 = f"{fj}-{fi}"
            rho = row.get(pair1, row.get(pair2, np.nan))
            if pd.isna(rho):
                rho = 0.0
            rho = float(np.clip(rho, -1.0 + 1e-6, 1.0 - 1e-6))
            R[i, j] = rho
            R[j, i] = rho
    return _ensure_psd(R)
