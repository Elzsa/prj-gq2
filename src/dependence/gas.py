# src/dependency_structure/gas.py

"""
Étape 3e — GAS (Generalized Autoregressive Score)
avec la skewed-t copula de Demarta & McNeil (2005)

Références :
    - Creal, Koopman & Lucas (2013). GAS models. J. Applied Econometrics.
    - Lucas, Schwaab & Zhang (2014). Conditional euro area sovereign default risk.
    - Demarta & McNeil (2005). The t copula and related copulas.
    - Salvatierra & Patton (2015). Dynamic copula models and high frequency data.

Optimisations implémentées (sans modifier le modèle) :
    1. Plage de dates (start_date / end_date) : permet de répartir le calcul
       entre plusieurs machines, chacune traitant une période différente.
    2. Reprise automatique (resume) : au démarrage, les dates déjà calculées
       (fichiers temporaires existants) sont skippées.
    3. Parallélisme par chunks : la plage de dates est divisée en N chunks
       (N = nb de cœurs). Chaque chunk tourne dans un sous-processus séparé
       avec warm starting interne. Gain : Nx plus rapide.
    4. Sauvegarde incrémentale : après chaque fenêtre, le résultat est écrit
       dans un fichier CSV temporaire. Si le processus plante, rien n'est perdu.

Le flux complet est :
uniforms
   ↓
rolling_gas()
   ↓
_process_chunk()
   ↓
pour chaque fenêtre de 60 mois :
    window = u_{t-60}, ..., u_{t-1}
       ↓
    R_bar = corrélation de Spearman de la fenêtre
       ↓
    _fit_gas_params()
       ↓
    _gas_neg_loglik()
       ↓
    _copula_ll_and_score()
       ↓
    _gas_step()
       ↓
    paramètres estimés : alpha, beta, nu, lambda
       ↓
    _compute_R_t()
       ↓
    dernière matrice R_t de la fenêtre
       ↓
stockage des 10 corrélations hors diagonale


Pipeline GAS — skewed-t copula

Étape 1 : Entrée du modèle
    On part des pseudo-uniformes u_{i,t} obtenus après transformation PIT
    des résidus standardisés issus des modèles AR-GJR-GARCH.
    Contrairement à DCC/ADCC, le GAS ne travaille donc pas directement sur
    les résidus z_{i,t}, mais sur les uniformes u_{i,t} ∈ (0,1).

Étape 2 : Fenêtre roulante
    Pour chaque date t, on extrait une fenêtre de 60 mois de pseudo-uniformes.
    Cette fenêtre sert à estimer localement la structure de dépendance entre
    les cinq facteurs Fama-French.

Étape 3 : Cible de long terme
    Sur chaque fenêtre, on calcule une matrice de corrélation inconditionnelle
    R_bar, utilisée comme cible de long terme dans la dynamique GAS.

Étape 4 : Estimation des paramètres GAS
    On estime par maximum de vraisemblance les paramètres :
        alpha  : réaction au score,
        beta   : persistance de la corrélation passée,
        nu     : degrés de liberté de la skewed-t,
        lambda : paramètre d'asymétrie.
    L'objectif est de trouver les paramètres qui maximisent la log-vraisemblance
    de la copule sur la fenêtre.

Étape 5 : Transformation des uniformes
    Pour évaluer la densité de copule, chaque vecteur u_t est transformé en
    quantiles skewed-t :
        x_t = F^{-1}(u_t ; nu, lambda).
    L'inversion de la CDF est faite numériquement sur une grille.

Étape 6 : Densité de copule
    La log-densité de copule est calculée comme :
        log c(u_t) = log f_N(x_t ; R_t, nu, lambda)
                     - somme_i log f_1(x_{i,t} ; nu, lambda).
    Elle mesure la vraisemblance de la dépendance observée à la date t.

Étape 7 : Calcul du score
    Le score est la dérivée de la log-densité de copule par rapport à la matrice
    de corrélation R_t. Il indique dans quelle direction ajuster R_t pour rendre
    l'observation plus vraisemblable.

Étape 8 : Mise à jour GAS de la corrélation
    La matrice de corrélation est mise à jour récursivement à partir :
        - de la cible de long terme R_bar,
        - du score courant,
        - de la corrélation passée.
    La mise à jour est effectuée dans l'espace arctanh afin de maintenir les
    corrélations dans l'intervalle (-1,1), puis la matrice est projetée pour
    rester une matrice de corrélation valide.

Étape 9 : Matrice finale de la fenêtre
    Une fois les paramètres estimés, on relance la récurrence GAS sur la fenêtre
    et on conserve uniquement la dernière matrice R_t. Elle représente la matrice
    de corrélation dynamique estimée pour la date courante.

Étape 10 : Sortie du modèle
    Pour chaque date, on stocke les 10 corrélations hors diagonale de la matrice
    5x5 sous forme de DataFrame :
        MKT_RF-SMB, MKT_RF-HML, ..., RMW-CMA.
    Ces matrices peuvent ensuite être comparées à DCC/ADCC ou utilisées dans
    la partie optimisation de portefeuille.

"""

import csv
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import gammaln, kve
from tqdm import tqdm

from config import config_dependance

# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — GH SKEWED-T UNIVARIÉE (Demarta & McNeil 2005)
# ══════════════════════════════════════════════════════════════════════════════


def _log_bessel(nu: float, x: float) -> float:
    """
    log K_ν(x) via la version scalée kve(ν,x) = exp(x)*K_ν(x).
        log K_ν(x) = log(kve(ν,x)) - x
    Numériquement stable pour x grands.
    """
    if x <= 0:
        return np.inf
    scaled = kve(nu, x)
    if scaled <= 0 or not np.isfinite(scaled):
        return -np.inf
    return float(np.log(scaled) - x)


def _gh_log_pdf(x: float, nu: float, lam: float) -> float:
    """
    Log-densité de la GH skewed-t univariée.

        log f(x; λ, ν) = log C_1
                        + log K_{(ν+1)/2}(|λ|·√(x²+ν))
                        + λ·x
                        - (ν+1)/4 · log(x²+ν)

        log C_1 = (1-ν/2)·log2 + (ν+1)/2·log|λ|
                  + ν/2·log(ν) - ½·log(2π) - logΓ(ν/2)
    """
    log_C = (
        (1 - nu / 2) * np.log(2)
        + (nu + 1) / 2 * np.log(abs(lam) + 1e-300)
        + nu / 2 * np.log(nu)
        - 0.5 * np.log(2 * np.pi)
        - gammaln(nu / 2)
    )
    arg = abs(lam) * np.sqrt(x**2 + nu)
    return (
        log_C
        + _log_bessel((nu + 1) / 2, arg)
        + lam * x
        - (nu + 1) / 4 * np.log(x**2 + nu)
    )


def _gh_cdf_grid(
    nu: float,
    lam: float,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    CDF de la GH skewed-t sur une grille par intégration numérique.

    Stratégie :
        1. Calculer la PDF sur x_grid
        2. Intégrer cumulativement (trapèzes)
        3. Ajouter la masse dans la queue gauche (x < x_grid[0])
        4. Normaliser à [0, 1]
    """
    # Calcul de la densité de la GH skewed-t en chaque point de x_grid
    pdf_vals = np.array([np.exp(_gh_log_pdf(xi, nu, lam)) for xi in x_grid])

    # Calcul une intégrale cumulative numérique de la densité pdf_vals sur la grille x_grid,
    # avec la méthode des trapèzes, de manière à obtenir la fonction de répartition.
    cdf = cumulative_trapezoid(pdf_vals, x_grid, initial=0.0)

    # NOTE : cumulative_trapezoid approxime l'intégrale de la densité seulement
    # à partir du premier point de la grille x_grid[0]. Or, par définition,
    # la CDF intègre la densité depuis -∞. On ajoute donc la masse de probabilité
    # située à gauche de x_grid[0] pour corriger ce décalage.
    # Correction : masse à gauche de x_grid[0]
    try:
        tail_left, _ = quad(
            lambda t: np.exp(_gh_log_pdf(t, nu, lam)),
            -np.inf,
            x_grid[0],
            limit=50,
            epsabs=1e-6,
        )
    except Exception:
        tail_left = 0.0

    # Ajout de la masse de queue gauche, puis sécurisation numérique :
    # une CDF doit rester dans [0, 1].
    cdf = np.clip(cdf + tail_left, 0.0, 1.0)

    # Le dernier point de la grille est forcé à 1 pour faciliter l'inversion
    # numérique de la CDF sur des pseudo-uniformes proches de 1.
    cdf[-1] = 1.0
    return cdf


def _gh_inv_cdf(
    u: float,
    x_grid: np.ndarray,
    cdf_grid: np.ndarray,
) -> float:
    """
    Retourne une approximation du quantile GH skewed-t F^{-1}(u).

    L'inversion est réalisée par interpolation linéaire à partir de la table
    numérique (x_grid, cdf_grid). Les points où la CDF est plate sont retirés
    afin d'obtenir une interpolation inverse stable, et u est borné pour éviter
    les problèmes numériques aux extrêmes.
    """
    # Assurer cdf strictement croissante
    idx = np.concatenate(([True], np.diff(cdf_grid) > 1e-12))
    cdf_clean = cdf_grid[idx]
    x_clean = x_grid[idx]

    inv = interp1d(
        cdf_clean,
        x_clean,
        kind="linear",
        fill_value=(x_clean[0], x_clean[-1]),
        bounds_error=False,
    )
    return float(inv(float(np.clip(u, cdf_clean[0] + 1e-9, cdf_clean[-1] - 1e-9))))


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — GH SKEWED-T MULTIVARIÉE ET DENSITÉ COPULE
# ══════════════════════════════════════════════════════════════════════════════


def _gh_joint_log_pdf(
    x: np.ndarray,
    R: np.ndarray,
    nu: float,
    lam: float,
) -> float:
    """
    Log-densité conjointe de la GH skewed-t multivariée (Σ=R, γ=λ·1).

        log f(x; R, λ, ν) = log C_N
                            + log K_{(ν+N)/2}(√(ψ_N·(q_N+ν)))
                            + λ·1'R⁻¹x
                            - (ν+N)/4 · log(q_N+ν)

        ψ_N  = λ²·(1'R⁻¹1)
        q_N  = x'R⁻¹x
        log C_N = (1-ν/2)·log2 + (ν+N)/2·log|λ|
                  + (ν+N)/4·log(1'R⁻¹1) + ν/2·log(ν)
                  - N/2·log(2π) - ½·log|R| - logΓ(ν/2)
    """
    N = len(x)
    ones = np.ones(N)

    # ── Guard numérique ───────────────────────────────────────────────────────
    # Vérification PD + log déterminant via Cholesky (fail-fast si non-DP)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        return -np.inf
    log_det_R = 2.0 * np.sum(np.log(np.diag(L)))

    if not np.isfinite(log_det_R):
        return -np.inf
    # ─────────────────────────────────────────────────────────────────────────

    # R⁻¹ via Cholesky
    L_inv = solve_triangular(L, np.eye(N), lower=True)
    R_inv = L_inv.T @ L_inv

    if not np.all(np.isfinite(R_inv)):
        return -np.inf

    ones_Rinv_ones = float(ones @ R_inv @ ones)
    q_N = float(x @ R_inv @ x)  # distance quadratique multivariée.

    # ── Guard numérique ───────────────────────────────────────────────────────
    # ones'R⁻¹ones > 0 est garanti si R est définie positive.
    # Si négatif → R est numériquement dégénérée → configuration invalide.
    if ones_Rinv_ones <= 0 or q_N < 0:
        return -np.inf
    # ─────────────────────────────────────────────────────────────────────────

    log_C_N = (
        (1 - nu / 2) * np.log(2)
        + (nu + N) / 2 * np.log(abs(lam) + 1e-300)
        + (nu + N) / 4 * np.log(ones_Rinv_ones)  # garanti > 0
        + nu / 2 * np.log(nu)
        - N / 2 * np.log(2 * np.pi)
        - 0.5 * log_det_R
        - gammaln(nu / 2)
    )

    arg_N = abs(lam) * np.sqrt(ones_Rinv_ones) * np.sqrt(q_N + nu)

    return (
        log_C_N
        + _log_bessel((nu + N) / 2, arg_N)
        + lam * float(ones @ R_inv @ x)
        - (nu + N) / 4 * np.log(q_N + nu)  # garanti > 0
    )


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 — SCORE NUMÉRIQUE ET RÉCURRENCE GAS
# ══════════════════════════════════════════════════════════════════════════════


def _project_to_correlation(M: np.ndarray) -> np.ndarray:
    """
    Projette une matrice sur l'espace des matrices de corrélation.
    Symétrie → valeurs propres positives → diagonale = 1.

    NOTE: Debug ok : Si M est une matrice de corrélation correcte, M et R sont identiques.
    """
    # S'assurer que M est symétrique
    M = (M + M.T) / 2
    # Décomposition spectrale de la matrice symétrique
    # ev = valeurs propres -> Permettent de vérifier que la matrice est définie positive
    # evec = vecteurs propres
    ev, evec = np.linalg.eigh(M)
    # Correction des valeurs propres négatives (sécurité pour garantir une matrice (quasi) définie positive)
    ev = np.maximum(ev, 1e-8)
    # Reconstruction de la matrice corrigée
    M = evec @ np.diag(ev) @ evec.T
    # Normalisation de la diagonale (sécurité)
    d = np.sqrt(np.diag(M))
    R = M / np.outer(d, d)
    # Forcer la diagonale à 1 (afin d'éviter les erreurs numériques résiduelles)
    np.fill_diagonal(R, 1.0)

    return R


def _gas_score_analytical(
    x_t: np.ndarray,
    R_t: np.ndarray,
    nu: float,
    lam: float,
) -> np.ndarray:
    """
    Score analytique GAS de la copule GH skewed-t w.r.t. R_t,
    scalé par l'information de Fisher (Lucas, Schwaab & Zhang 2014).

    Gradient analytique :
        ∇_R = -½R⁻¹ + w_r·rr' + w_g·gg' - λ/2·(gr'+rg')

    où :
        r   = R⁻¹x                         projection de x
        g   = R⁻¹1                         projection du vecteur unité
        q   = r'x                          distance de Mahalanobis²
        ψ   = λ²·(1'g)                     contribution de l'asymétrie
        a   = √(ψ·(q+ν))                   argument des fonctions de Bessel
        κ   = (ν+N)/2                      ordre des Bessel
        d_K = -(K_{κ-1}(a) + K_{κ+1}(a)) / (2K_κ(a))   dérivée log K_κ

        w_r = (ν+N)/(4(q+ν)) − d_K·ψ/(2a)
        w_g = −(ν+N)/(4·(1'g)) − d_K·λ²·(q+ν)/(2a)

    Scaling Fisher information (Lucas et al. 2014, adapté GH) :
        s_t = 2(q+ν)² / ((ν+N)(ν+N+2)) · ∇_R

    Scaling inspiré du Fisher information scaling utilisé dans la littérature GAS.

    Paramètres
    ──────────
    x_t : np.ndarray — (N,) quantiles GH (déjà transformés depuis u_t)
    R_t : np.ndarray — (N×N) matrice de corrélation courante
    nu  : float      — degrés de liberté
    lam : float      — asymétrie scalaire

    Retourne
    ────────
    s_t : np.ndarray — (N×N) score GAS scalé, symétrique, diag=0
    """
    N = len(x_t)
    R_inv = np.linalg.inv(R_t)
    ones = np.ones(N)

    # Vecteurs de projection
    r = R_inv @ x_t  # R⁻¹x
    g = R_inv @ ones  # R⁻¹1
    q = float(x_t @ r)  # x'R⁻¹x  (Mahalanobis²)

    # Termes liés à l'asymétrie
    ones_g = float(ones @ g)  # 1'R⁻¹1
    psi = lam**2 * ones_g  # λ²·(1'R⁻¹1)

    psi_q = max(psi * (q + nu), 1e-300)

    a = np.sqrt(psi_q)  # argument Bessel

    kappa = (nu + N) / 2

    # Ratio de Bessel via version scalée (stable numériquement)
    # kve(ν, x) = exp(x)·K_ν(x)  →  K_{κ-1}/K_κ = kve(κ-1,a)/kve(κ,a)
    kve_k = kve(kappa, a)
    kve_km1 = kve(kappa - 1, a)
    kve_kp1 = kve(kappa + 1, a)

    if kve_k <= 0 or not np.isfinite(kve_k):
        return np.zeros((N, N))  # configuration invalide → score nul

    # Calcul de la dérivée logarithmique de la fonction de Bessel
    d_K = -(kve_km1 + kve_kp1) / (2.0 * kve_k)

    # Poids des termes outer-products
    lam2 = lam**2
    w_r = (nu + N) / (4.0 * (q + nu)) - d_K * psi / (2.0 * a)
    w_g = -(nu + N) / (4.0 * max(ones_g, 1e-300)) - d_K * lam2 * (q + nu) / (2.0 * a)

    # Gradient analytique ∇_R
    grad = (
        -0.5 * R_inv
        + w_r * np.outer(r, r)
        + w_g * np.outer(g, g)
        - lam / 2.0 * (np.outer(g, r) + np.outer(r, g))
    )

    # Symétrie numérique + diag=0 (R a des 1 sur la diagonale)
    grad = (grad + grad.T) / 2.0
    np.fill_diagonal(grad, 0.0)

    # Scaling Fisher information (Lucas et al. 2014)
    fisher_scale = 2.0 * (q + nu) ** 2 / ((nu + N) * (nu + N + 2.0))
    s_t = fisher_scale * grad

    return s_t


def _copula_ll_and_score(
    u_t: np.ndarray,
    R_t: np.ndarray,
    nu: float,
    lam: float,
    x_grid: np.ndarray,
    cdf_grid: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Calcule la log-densité copule ET le score analytique en un seul appel.

    Avantage : l'inversion CDF (u → x) n'est faite qu'une seule fois,
    puis réutilisée pour le score — évite le double calcul.

    Retourne
    ────────
    (ll_t, s_t) : (float, np.ndarray)
    """
    N = len(u_t)  # N = Nombre de facteurs

    # Transformation des uniformes en quantiles skewed-t
    # NOTE : Cette étape est importante : La copule est définie sur les uniformes, mais la
    # densité skewed-t multivariée s’évalue sur les quantiles x_t.
    x_t = np.array([_gh_inv_cdf(u_t[i], x_grid, cdf_grid) for i in range(N)])

    # Calcul de la log densité jointe multivariée
    # Elle mesure la vraisemblance du vecteur complet x_t, en tenant compte de la dépendance entre les facteurs.
    log_f_multi = _gh_joint_log_pdf(x_t, R_t, nu, lam)

    # Somme des log densités marginales
    # On mesure la vraisemblance de chaque facteur séparément, sans la dépendance
    log_f_uni = sum(_gh_log_pdf(x_t[i], nu, lam) for i in range(N))

    # NOTE : densité de copule = densité jointe - effet des marginales
    ll_t = float(log_f_multi - log_f_uni)

    # Calcul du score GAS
    s_t = _gas_score_analytical(x_t, R_t, nu, lam)

    return ll_t, s_t


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 4 — MLE DES PARAMÈTRES GAS
# ══════════════════════════════════════════════════════════════════════════════


def _gas_step(
    R_t: np.ndarray,
    s_t: np.ndarray,
    R_bar: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Effectue un pas de la récurrence GAS en espace arctanh (Fisher z-transform).

    Paramétrage arctanh (Salvatierra & Patton 2015) :
        z_{ij,t}   = arctanh(R_{ij,t})          espace non borné
        s^z_{ij,t} = s^R_{ij,t} · (1-R²_{ij,t}) score en espace z (chain rule)
        z_{t+1}    = (1-α-β)·z_bar + α·s^z_t + β·z_t
        R_{t+1}    = tanh(z_{t+1})               retour dans (-1,+1)

    Avantage : tanh borne naturellement chaque entrée dans (-1,+1).

    Cette paramétisation est conforme à la littérature GAS pour les copules
    (Salvatierra & Patton 2015, cité dans Zhao et al. 2019).
    """
    N = R_t.shape[0]

    # Pré-clipping léger pour garantir arctanh est défini (évite ±1 exacts)
    R_safe = np.clip(R_t, -0.9999, 0.9999)
    Rb_safe = np.clip(R_bar, -0.9999, 0.9999)

    # Transformation vers espace non borné
    z_t = np.arctanh(R_safe)
    z_bar = np.arctanh(Rb_safe)

    # Score en espace z via chain rule : s^z = s^R · (1 - R²)
    s_z = s_t * (1.0 - R_safe**2)
    np.fill_diagonal(s_z, 0.0)  # la diagonale de R est fixée à 1

    # Récurrence GAS en espace z
    z_new = (1.0 - alpha - beta) * z_bar + alpha * s_z + beta * z_t
    # Sécurité numérique : éviter la saturation de tanh (tanh(±3) ≈ ±0.995)
    z_new = np.clip(z_new, -3.0, 3.0)

    # Retour dans l'espace des corrélations via tanh
    R_new = np.tanh(z_new)
    np.fill_diagonal(R_new, 1.0)

    # Projection finale pour garantir définie-positivité
    R_new = _project_to_correlation(R_new)

    return R_new


def _gas_neg_loglik(params, u, R_bar, x_grid):
    """
    Calcule la log-vraisemblance négative du modèle GAS-skewed-t sur une fenêtre.

    Pour un vecteur de paramètres candidat (alpha, beta, nu, lambda), la fonction
    évalue la vraisemblance de la copule GH skewed-t dynamique sur les
    pseudo-uniformes u. La matrice de corrélation est initialisée à R_bar, puis
    mise à jour récursivement à chaque observation à partir du score de la
    log-densité de copule.

    La fonction retourne -ll car l'optimiseur scipy minimise l'objectif. Les
    configurations non stationnaires ou numériquement instables sont pénalisées
    par une grande valeur de retour.
    """
    alpha, beta, nu, lam = params
    T, _ = u.shape

    # Pénalités numériques : on rejette les paramètres qui rendent la dynamique
    # quasi non stationnaire ou qui placent la skewed-t dans une zone instable.
    if alpha + beta >= 0.9999:
        return 1e10

    if abs(lam) > 0.35 or abs(lam) < 0.01:
        return 1e10

    try:
        # Construction de la CDF de la GH skewed-t sur la grille x_grid
        cdf_grid = _gh_cdf_grid(nu, lam, x_grid)
    except Exception:
        return 1e10

    # Initialisation de la matrice dynamique
    R_t = R_bar.copy()
    # Initialisation de la log-vraisemblance à 0
    ll = 0.0

    # NOTE : Choix d'implémentation — on tolère un faible nombre d'accidents
    # numériques locaux afin de ne pas rejeter un jeu de paramètres pour une
    # instabilité ponctuelle de la densité ou du score.
    n_failures = 0
    max_failures = max(3, int(0.05 * T))

    for t in range(T):
        try:
            # Calcul de la log-densité de copule à la date t (l_t) et du score GAS (s_t)
            # NOTE : le score GAS est l'information utilisée pour mettre à jour la matrice de corrélation R_t
            ll_t, s_t = _copula_ll_and_score(u[t], R_t, nu, lam, x_grid, cdf_grid)
        except Exception:
            n_failures += 1
            R_t = R_bar.copy()  # reset et continue
            continue

        # Vérification que la log-vraisemblance est valide
        if not np.isfinite(ll_t):
            n_failures += 1
            R_t = R_bar.copy()  # reset et continue (pas de return 1e10)
            continue

        ll += ll_t
        # Mise à jour de R_t
        R_t = _gas_step(R_t, s_t, R_bar, alpha, beta)

    if n_failures > max_failures:
        return 1e10

    return -ll


def _fit_gas_params(u, R_bar, x_grid, x0=None, date_str: str | None = None):
    """
    Estime les paramètres du modèle GAS-skewed-t par maximum de vraisemblance.

    Pour une fenêtre donnée de pseudo-uniformes PIT, la fonction cherche les
    paramètres de la dynamique GAS et de la copule GH skewed-t :

        theta = (alpha, beta, nu, lambda)

    où :
        alpha  : sensibilité de la matrice de corrélation au score courant ;
        beta   : persistance de la matrice de corrélation passée ;
        nu     : degrés de liberté de la skewed-t, contrôlant l'épaisseur
                 des queues ;
        lambda : paramètre d'asymétrie de la skewed-t.

    L'estimation minimise la log-vraisemblance négative de la copule sur la
    fenêtre, ce qui revient à maximiser la vraisemblance du modèle.

    La procédure d'optimisation se fait en deux temps :
        1. SLSQP est utilisé en première tentative, car il permet de gérer
           directement les bornes et la contrainte alpha + beta < 1.
        2. Si SLSQP échoue ou si le point initial est invalide, une seconde
           tentative est réalisée avec Nelder-Mead, à partir d'un point initial
           plus persistant et proche d'une dynamique quasi statique.

    Des bornes sont imposées aux paramètres afin de stabiliser l'estimation
    numérique et d'éviter les zones où la densité skewed-t ou la récurrence GAS
    deviennent instables. La contrainte alpha + beta < 1 garantit une dynamique
    stationnaire / non explosive.

    Si les deux optimisations échouent, la fonction retourne None. La fonction
    appelante utilisera alors la matrice de corrélation inconditionnelle R_bar
    comme fallback pour la fenêtre considérée.

    Paramètres
    ----------
    u : np.ndarray
        Fenêtre de pseudo-uniformes PIT de taille (T, N), où T est la taille de
        la fenêtre roulante et N le nombre de facteurs.

    R_bar : np.ndarray
        Matrice de corrélation cible de long terme, généralement calculée comme
        la corrélation de Spearman des pseudo-uniformes sur la fenêtre.

    x_grid : np.ndarray
        Grille numérique utilisée pour approximer la CDF et l'inverse CDF de la
        GH skewed-t.

    x0 : np.ndarray | None, optionnel
        Point de départ de l'optimisation. Si None, un point initial par défaut
        est utilisé.

    date_str : str | None, optionnel
        Date associée à la fenêtre courante, utilisée uniquement pour améliorer
        les messages de log en cas d'échec de convergence.

    Retourne
    --------
    np.ndarray | None
        Vecteur estimé (alpha, beta, nu, lambda) si l'optimisation converge,
        sinon None.
    """
    if x0 is None:
        # Valeur initiale
        x0 = np.array([0.003, 0.88, 9.0, -0.28])

    # Intervalles de recherche des paramètres
    bounds = [(1e-6, 0.01), (1e-6, 0.9999), (2.1, 30.0), (-0.4, 0.4)]
    # NOTE : # La borne supérieure de alpha est volontairement restrictive.
    # Elle a été fixée empiriquement afin de stabiliser la récurrence GAS :
    # des valeurs plus élevées entraînaient des réactions excessives au score
    # et pouvaient produire des matrices de corrélation mal conditionnées.

    # Contrainte de stationnarité
    constraint = {"type": "ineq", "fun": lambda x: 0.9999 - x[0] - x[1]}

    # Étape 1 : SLSQP si le x0 est dans une zone valide (rapide, précis)
    if _gas_neg_loglik(x0, u, R_bar, x_grid) < 1e6:
        result = minimize(
            _gas_neg_loglik,
            x0,
            args=(u, R_bar, x_grid),
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"ftol": 1e-5, "maxiter": 200},
        )
        if result.success:
            return result.x

    # Étape 2 : Nelder-Mead near-static pour les fenêtres extrêmes
    # (gradient plat au départ → SLSQP échouerait, Nelder-Mead explore sans gradient)

    # Point initial inspiré d'implémentations de copules dynamiques de type Patton :
    # alpha faible et beta élevé correspondent à une dynamique proche du cas statique,
    # ce qui fournit un point de départ robuste lorsque la surface de vraisemblance
    # est plate ou instable.
    x0_fallback = np.array([0.001, 0.989, 8.0, -0.1])
    result = minimize(
        _gas_neg_loglik,
        x0_fallback,
        args=(u, R_bar, x_grid),
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000},
    )
    if result.success and result.fun < 1e6:
        a, b = result.x[0], result.x[1]
        # Scipy ne supporte pas les contraintes pour Nelder-Mead
        # Donc on ne peut pas passer constraints=[constraint] dans minimize(...)
        # On vérifie donc manuellement la contrainte de stationnarité
        if a + b < 0.9999:
            return result.x

    logging.warning(
        f"[WARN] GAS MLE : convergence échouée pour {date_str} — R_bar utilisé"
    )
    return None


def _compute_R_t(
    u: np.ndarray,
    params: np.ndarray,
    R_bar: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    Calcule R_T (dernière R de la récurrence GAS) pour une fenêtre donnée.

    Utilise le score analytique (Fisher information scaling) et la
    récurrence en espace arctanh (Salvatierra & Patton 2015) pour
    garantir la stabilité — conforme à Creal et al. (2013).
    """
    alpha, beta, nu, lam = params
    N = u.shape[1]
    cdf_grid = _gh_cdf_grid(nu, lam, x_grid)
    R_t = R_bar.copy()

    for t in range(len(u)):
        x_t = np.array([_gh_inv_cdf(u[t][i], x_grid, cdf_grid) for i in range(N)])
        s_t = _gas_score_analytical(x_t, R_t, nu, lam)
        R_t = _gas_step(R_t, s_t, R_bar, alpha, beta)

    return R_t


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 5 — TRAITEMENT D'UN CHUNK (fonction appelée par chaque sous-processus)
# ══════════════════════════════════════════════════════════════════════════════

# Grille de valeurs x utilisée pour approximer numériquement la CDF
# et l'inverse CDF de la GH skewed-t.
# La grille est plus dense autour de 0, où se concentre la majorité
# de la masse de probabilité, et plus espacée dans les queues.
# En résumé, _X_GRID sert à construire une approximation numérique de F^(−1) à partir de F.
_X_GRID = np.concatenate(
    [
        np.linspace(-20, -5, 50),  # queue gauche
        np.linspace(-5, 5, 200),  # zone centrale, plus dense
        np.linspace(5, 20, 50),  # queue droite
    ]
)


def _process_chunk(args: tuple) -> str:
    """
    Traite un chunk de dates, chaque fenêtre indépendamment.

    Appelé dans un sous-processus par multiprocessing.Pool.
    Sauvegarde chaque résultat immédiatement dans un fichier CSV temporaire.

    Paramètres (via args tuple — nécessaire pour multiprocessing)
    ─────────────────────────────────────────────────────────────
    uniforms_array  : np.ndarray  — (T x N) pseudo-uniformes
    all_dates_str   : list[str]   — dates de uniforms_array (format ISO)
    t_indices       : list[int]   — indices dans uniforms_array à traiter
    window_size     : int
    x_grid          : np.ndarray
    tmp_file        : str         — chemin du fichier CSV temporaire
    x0_init         : np.ndarray  — ignoré (conservé pour compatibilité args)
    pairs           : list[str]   — noms des paires de facteurs

    Retourne
    ────────
    tmp_file : str — chemin du fichier CSV complété
    """
    (
        uniforms_array,
        all_dates_str,
        t_indices,  # NOTE : t_indices = chuck
        window_size,
        x_grid,
        tmp_file,
        _,  # x0_init ignoré (plus de warm starting)
        pairs,
    ) = args

    N = uniforms_array.shape[1]
    tmp_path = Path(tmp_file)

    # Charger les dates déjà calculées dans ce chunk (reprise automatique)
    done_dates: set[str] = set()
    if tmp_path.exists():
        try:
            done_df = pd.read_csv(tmp_path)
            done_dates = set(done_df["date"].astype(str).tolist())
        except Exception:
            pass

    for t_idx in t_indices:

        date_str = str(all_dates_str[t_idx])[:10]  # "YYYY-MM-DD"

        # Reprise : skip si déjà calculé
        if date_str in done_dates:
            continue

        # Fenêtre de 60 mois strictement avant t
        window = uniforms_array[t_idx - window_size : t_idx]

        # Corrélation de Spearman comme cible inconditionnelle (variance targeting)

        # Calcule la matrice de corrélation de Spearman des pseudo-uniformes
        # sur la fenêtre roulante. Spearman mesure la dépendance entre les rangs,
        # ce qui est cohérent avec l'approche copule.
        # Cette matrice sert de cible de long terme R_bar dans la dynamique GAS.
        R_bar = np.array(pd.DataFrame(window).corr(method="spearman"))

        # Projection de sécurité : garantit que R_bar est bien une matrice
        # de corrélation valide symétrique, définie positive, avec diagonale égale à 1.
        R_bar = _project_to_correlation(R_bar)

        # ── MLE — fresh start à chaque fenêtre ───────────────────────────────
        try:
            params = _fit_gas_params(window, R_bar, x_grid, x0=None, date_str=date_str)
        except Exception as e:
            logging.warning(f"[WARN] GAS MLE échouée pour {date_str} : {e}")
            params = None

        # ── Calcul de R_t ─────────────────────────────────────────────────────
        if params is None:
            # MLE échouée → fallback corrélation inconditionnelle
            R_t = R_bar.copy()
        else:
            try:
                R_t = _compute_R_t(window, params, R_bar, x_grid)
            except Exception as e:
                logging.warning(f"[WARN] GAS R_t échoué pour {date_str} : {e}")
                R_t = R_bar.copy()

        # Détecter les matrices R_t suspectes : si toutes les corrélations hors diagonale
        # ont presque la même magnitude, la dynamique GAS a probablement produit une
        # matrice quasi plate ou numériquement peu informative. Dans ce cas, on revient
        # à la corrélation inconditionnelle R_bar de la fenêtre.
        offdiag = np.abs(R_t[np.triu_indices(N, k=1)])
        if np.std(offdiag) < 0.02:  # toutes les corrélations identiques → dégénérée
            logging.warning(f"[WARN] R_T quasi plate pour {date_str} — R_bar utilisé")
            R_t = R_bar.copy()

        # ── Sauvegarde incrémentale ───────────────────────────────────────────
        row: dict = {"date": date_str}
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                row[pairs[k]] = float(R_t[i, j])
                k += 1

        file_exists = tmp_path.exists()
        with open(tmp_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date"] + pairs)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    return str(tmp_file)


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 6 — FONCTION PRINCIPALE (ROLLING WINDOW AVEC TOUTES LES OPTIMISATIONS)
# ══════════════════════════════════════════════════════════════════════════════


def rolling_gas(
    uniforms: pd.DataFrame,
    window_size: int = config_dependance.WINDOW_SIZE,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: Path = config_dependance.RESULTS_DIR,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Applique le modèle GAS (copule GH skewed-t) avec fenêtre roulante.

    Optimisations incluses (sans modifier le modèle) :
        - Plage de dates : start_date / end_date pour répartir entre machines
        - Reprise automatique : skips les dates déjà calculées
        - Chunks parallèles : N chunks x N workers

    Paramètres
    ──────────
    uniforms    : pd.DataFrame — pseudo-uniformes u_{i,t} (Tx5) de garch.py
    window_size : int          — fenêtre roulante en mois (défaut : 60)
    start_date  : str | None   — début de la plage à calculer (ex: "2000-01")
                                 None = début de la période out-of-sample
    end_date    : str | None   — fin de la plage (ex: "2004-11")
                                 None = fin des données disponibles
    output_dir  : Path         — dossier de sortie pour les fichiers temporaires
    n_workers   : int | None   — nombre de workers (None = os.cpu_count())

    Retourne
    ────────
    correlations : pd.DataFrame — matrices R_t sérialisées
                                  index = dates, colonnes = paires de facteurs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factors = uniforms.columns.tolist()
    N = len(factors)
    pairs = [f"{factors[i]}-{factors[j]}" for i in range(N) for j in range(i + 1, N)]

    # ── 1. Filtrage de la plage de dates ─────────────────────────────────────
    if start_date is None:
        start_date = config_dependance.OUT_SAMPLE_START
    if end_date is None:
        end_date = uniforms.index[-1].strftime("%Y-%m")

    # Masque booléen : True pour les lignes à garder, False sinon.
    mask = (uniforms.index >= pd.Timestamp(start_date)) & (
        uniforms.index <= pd.Timestamp(end_date)
    )
    # On récupère les dates à garder
    target_dates = uniforms.index[mask]

    if len(target_dates) == 0:
        logging.warning("[WARN] GAS : aucune date dans la plage demandée.")
        return pd.DataFrame(columns=pairs)

    # Indices dans le DataFrame complet
    # Associe chaque date de l'index à sa position numérique dans le DataFrame complet.
    # Ces indices sont nécessaires pour construire les fenêtres roulantes sur l'array NumPy.
    all_indices = {d: i for i, d in enumerate(uniforms.index)}
    # Convertit les dates cibles en indices numériques et garde seulement celles
    # pour lesquelles on dispose d'au moins window_size observations passées.
    t_indices = [all_indices[d] for d in target_dates if all_indices[d] >= window_size]
    # Convertit toutes les dates en chaînes "YYYY-MM-DD", utilisées ensuite
    # pour nommer les fichiers temporaires et écrire les dates dans les CSV.
    all_dates_str = [str(d)[:10] for d in uniforms.index]

    logging.info(
        f"GAS : {len(t_indices)} fenêtres à calculer " f"({start_date} → {end_date})"
    )

    # ── 2. Détection des workers ──────────────────────────────────────────────
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, len(t_indices))
    n_workers = max(1, n_workers)

    logging.info(f"GAS : {n_workers} worker(s)")

    # ── 3. Division en chunks ─────────────────────────────────────────────────
    chunk_size = max(1, len(t_indices) // n_workers)
    chunks = [
        t_indices[i * chunk_size : (i + 1) * chunk_size] for i in range(n_workers - 1)
    ]
    # Dernier chunk prend le reste
    last_start = (n_workers - 1) * chunk_size
    chunks.append(t_indices[last_start:])
    chunks = [c for c in chunks if len(c) > 0]

    # ── 4. Préparation des arguments pour chaque chunk ────────────────────────

    # Convertit les pseudo-uniformes en array NumPy, plus simple à transmettre
    # aux sous-processus que le DataFrame pandas complet.
    uniforms_array = uniforms.values.astype(float)

    # Point de départ par défaut pour les paramètres GAS :
    # alpha, beta, nu, lambda.
    # Conservé dans les arguments même si le warm starting n'est plus utilisé.
    x0_default = np.array([0.05, 0.90, 8.0, 0.1])

    # Prépare la liste des arguments à envoyer à chaque chunk/worker.
    chunk_args = []
    for idx, chunk in enumerate(chunks):
        # Identifie la première et la dernière date du chunk afin de créer
        # un fichier temporaire unique pour cette plage de calcul.
        chunk_start = all_dates_str[chunk[0]].replace("-", "")[:6]
        chunk_end = all_dates_str[chunk[-1]].replace("-", "")[:6]
        tmp_file = output_dir / f"gas_tmp_{chunk_start}_{chunk_end}.csv"

        # Chaque chunk reçoit toutes les informations nécessaires pour calculer
        # ses fenêtres de manière indépendante et sauvegarder ses résultats.
        chunk_args.append(
            (
                uniforms_array,
                all_dates_str,
                chunk,
                window_size,
                _X_GRID,
                str(tmp_file),
                x0_default,
                pairs,
            )
        )

    # ── 5. Lancement parallèle des chunks ─────────────────────────────────────
    if n_workers == 1:
        # Mode séquentiel (pas de multiprocessing) — utile pour debug
        tmp_files = [_process_chunk(chunk_args[0])]
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            tmp_files = list(
                tqdm(
                    pool.imap(_process_chunk, chunk_args),
                    total=len(chunk_args),
                    desc="GAS chunks",
                    unit="chunk",
                )
            )

    # ── 6. Fusion des fichiers temporaires ────────────────────────────────────
    frames = []
    for tmp_file in tmp_files:
        p = Path(tmp_file)
        if p.exists() and p.stat().st_size > 0:
            try:
                df_tmp = pd.read_csv(p, parse_dates=["date"])
                df_tmp = df_tmp.set_index("date")
                frames.append(df_tmp)
            except Exception as e:
                logging.warning(f"[WARN] Impossible de lire {tmp_file} : {e}")

    if not frames:
        return pd.DataFrame(columns=pairs)

    correlations = pd.concat(frames).sort_index()
    correlations = correlations[~correlations.index.duplicated(keep="first")]

    logging.info(f"GAS : {len(correlations)} matrices R_t calculées")

    return correlations


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 7 — FUSION DES RÉSULTATS DE PLUSIEURS MACHINES
# ══════════════════════════════════════════════════════════════════════════════


def merge_gas_results(
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """
    Fusionne tous les fichiers CSV temporaires GAS d'un dossier.

    Permet d'obtenir un fichier unique

    Paramètres
    ──────────
    output_dir : Path — dossier contenant les fichiers gas_tmp_*.csv
    fmt        : str  — format de sortie final ('parquet' ou 'xlsx')

    Retourne
    ────────
    correlations : pd.DataFrame — résultats fusionnés et triés
    """
    output_dir = Path(output_dir)
    tmp_files = sorted(output_dir.glob("gas_tmp_*.csv"))

    if not tmp_files:
        logging.warning("[WARN] merge_gas_results : aucun fichier gas_tmp_*.csv trouvé")
        return pd.DataFrame()

    frames = []
    for f in tmp_files:
        try:
            # parse_dates convertit automatiquement la colonne "date" au format date
            df = pd.read_csv(f, parse_dates=["date"])
            # On met la colonne "date" en index du DataFrame
            df = df.set_index("date")
            frames.append(df)
            logging.info(f"  Chargé : {f.name} ({len(df)} lignes)")
        except Exception as e:
            logging.warning(f"[WARN] Impossible de lire {f.name} : {e}")

    correlations = pd.concat(frames).sort_index()
    # Suppression des dates dupliquées
    correlations = correlations[~correlations.index.duplicated(keep="first")]

    # Sauvegarde du fichier final
    if fmt == "parquet":
        path = output_dir / "correlations_gas.parquet"
        correlations.to_parquet(path)
    elif fmt == "xlsx":
        path = output_dir / "correlations_gas.xlsx"
        correlations.to_excel(path)
    else:
        raise ValueError(f"Format non supporté : '{fmt}'.")

    logging.info(f"GAS fusionné ({len(correlations)} lignes) → {path}")
    return correlations


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 8 — UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════


def get_matrix_at(correlations: pd.DataFrame, date, factors: list) -> np.ndarray:
    """Reconstruit la matrice de corrélation 5x5 complète à une date donnée."""
    N = len(factors)
    # Créer une matrice identité de taille NxN
    R = np.eye(N)
    row = correlations.loc[date]
    for i in range(N):
        for j in range(i + 1, N):
            pair = f"{factors[i]}-{factors[j]}"
            R[i, j] = R[j, i] = float(row[pair])
    return R


def save_gas(
    correlations: pd.DataFrame,
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> Path:
    """Sauvegarde les corrélations GAS dans un fichier final."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        path = output_dir / "correlations_gas.parquet"
        correlations.to_parquet(path)
    elif fmt == "xlsx":
        path = output_dir / "correlations_gas.xlsx"
        correlations.to_excel(path)
    else:
        raise ValueError(f"Format non supporté : '{fmt}'.")
    logging.info(f"GAS corrélations sauvegardées → {path}")
    return path


def load_gas(
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """Charge les corrélations GAS depuis un fichier sauvegardé."""
    output_dir = Path(output_dir)
    if fmt == "parquet":
        return pd.read_parquet(output_dir / "correlations_gas.parquet")
    elif fmt == "xlsx":
        df = pd.read_excel(output_dir / "correlations_gas.xlsx", index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    else:
        raise ValueError(f"Format non supporté : '{fmt}'.")


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 9 — FONCTION DE DEBUG
# ══════════════════════════════════════════════════════════════════════════════
def _copula_log_density(
    u_t: np.ndarray,
    R_t: np.ndarray,
    nu: float,
    lam: float,
    x_grid: np.ndarray,
    cdf_grid: np.ndarray,
) -> float:
    """
    Log-densité de la copule GH skewed-t à l'instant t.

        log c_t = log f(x_t; R_t, λ, ν) - Σ_i log f_i(x_{i,t}; λ, ν)

    où x_{i,t} = F_i⁻¹(u_{i,t}) via interpolation sur grille.
    """
    N = len(u_t)
    x_t = np.array([_gh_inv_cdf(u_t[i], x_grid, cdf_grid) for i in range(N)])

    log_f_multi = _gh_joint_log_pdf(x_t, R_t, nu, lam)
    log_f_uni = sum(_gh_log_pdf(x_t[i], nu, lam) for i in range(N))

    return float(log_f_multi - log_f_uni)
