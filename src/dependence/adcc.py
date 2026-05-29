# src/dependency_structure/adcc.py

"""
Étape 3d — Modèle ADCC (Asymmetric Dynamic Conditional Correlation)
Cappiello, Engle & Sheppard (2006)

Input  : résidus standardisés z_{i,t} issus de garch.py
Output : matrices de corrélation dynamiques R_t (5×5) pour chaque mois t

Différence avec DCC :
    Le DCC traite les chocs positifs et négatifs de la même façon.
    L'ADCC ajoute un terme asymétrique : les chocs NÉGATIFS augmentent
    davantage les corrélations que les chocs positifs (effet de levier
    multivarié — les marchés ont tendance à chuter ensemble).

Dynamique ADCC (version scalaire) :
    n_t   = z_t ⊙ 1[z_t < 0]           (partie négative des résidus)
    N̄     = E[n_t n'_t]                  (moyenne inconditionnelle)
    Q_t   = (1-a-b)Q̄ - g*N̄ + a*z_{t-1}*z'_{t-1} + g*n_{t-1}*n'_{t-1} + b*Q_{t-1}
    R_t   = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

Paramètres :
    a : réactivité aux chocs (comme DCC)
    b : persistance des corrélations (comme DCC)
    g : asymétrie — réactivité supplémentaire aux chocs négatifs

Si g = 0, ADCC = DCC.

Contrainte de stationnarité : a + b + g < 1

Référence : Cappiello, L., Engle, R. F., & Sheppard, K. (2006).
            Asymmetric dynamics in the correlations of global equity and bond returns.
            Journal of Financial Econometrics, 4(4), 537-572.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from config import config_dependance

# ── Log-vraisemblance ADCC ────────────────────────────────────────────────────


def _adcc_loglikelihood(params: np.ndarray, z: np.ndarray) -> float:
    """
    Calcule la log-vraisemblance négative du modèle ADCC.

    Identique à DCC avec le terme asymétrique supplémentaire :
        l_t = -1/2 [log|R_t| + z_t' R_t⁻¹ z_t - z_t' z_t]

    Paramètres
    ----------
    params : np.ndarray — [a, b, g]
    z      : np.ndarray — (T x N) résidus standardisés

    Retourne
    --------
    neg_ll : float — log-vraisemblance négative (à minimiser)
    """
    a, b, g = params
    T, N = z.shape

    # Covariance empirique Q̄
    Q_bar = np.cov(z.T)
    # Normalisation pour obtenir une corrélation
    D = np.sqrt(np.diag(Q_bar))
    Q_bar = Q_bar / np.outer(D, D)

    # Partie négative des résidus : n_t = z_t ⊙ 1[z_t < 0]
    # NOTE : # On isole les chocs négatifs car l'ADCC cherche à capter une asymétrie
    # dans la dynamique des corrélations : en période de stress, les mauvaises
    # surprises simultanées peuvent renforcer la dépendance entre les facteurs.
    # Les chocs positifs sont donc mis à zéro, seuls les résidus négatifs sont
    # conservés pour construire le terme asymétrique n_t n_t'.
    n = z * (z < 0)

    # Moyenne inconditionnelle N̄ = E[n_t n'_t]
    N_bar = (n.T @ n) / T

    Q = Q_bar.copy()
    # Initialisation de la log-vraisemblance à 0
    ll = 0.0

    for t in range(1, T):
        z_prev = z[t - 1].reshape(-1, 1)  # N × 1
        n_prev = n[t - 1].reshape(-1, 1)  # N × 1

        # Récurrence ADCC
        Q = (
            (1 - a - b) * Q_bar
            - g * N_bar
            + a * (z_prev @ z_prev.T)
            + g * (n_prev @ n_prev.T)
            + b * Q
        )

        # Normalisation → matrice de corrélation R_t
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)

        # Contribution à la log-vraisemblance
        z_t = z[t]
        # Calcul du déterminant
        sign, log_det = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10
        R_inv = np.linalg.inv(R)

        ll += -0.5 * (log_det + z_t @ R_inv @ z_t - z_t @ z_t)

    return -ll


# ── Estimation des paramètres ─────────────────────────────────────────────────


def _fit_adcc_params(z: np.ndarray) -> tuple[float, float, float]:
    """
    Estime les paramètres (a, b, g) de l'ADCC par MLE sur une fenêtre.

    Initialisation : on part des valeurs DCC typiques et on ajoute g=0.05.
    Contrainte     : a + b + g < 1  (stationnarité)
    Bornes         : a, b, g > 0

    Paramètres
    ----------
    z : np.ndarray — (T x N) résidus standardisés sur la fenêtre

    Retourne
    --------
    (a, b, g) : tuple[float, float, float]
    """
    result = minimize(
        fun=_adcc_loglikelihood,
        x0=np.array([0.05, 0.85, 0.05]),
        args=(z,),
        method="SLSQP",
        bounds=[(1e-6, 0.5), (1e-6, 0.9999), (1e-6, 0.5)],
        constraints=[{"type": "ineq", "fun": lambda x: 0.9999 - x[0] - x[1] - x[2]}],
        options={"ftol": 1e-8, "maxiter": 500},
    )
    if result.success:
        return float(result.x[0]), float(result.x[1]), float(result.x[2])
    else:
        logging.warning(
            "[WARN] ADCC : convergence échouée — valeurs initiales utilisées"
        )
        return 0.05, 0.85, 0.05


# ── Calcul de R_t sur une fenêtre ─────────────────────────────────────────────


def _compute_R_t(z: np.ndarray, a: float, b: float, g: float) -> np.ndarray:
    """
    Calcule la matrice de corrélation R_t au DERNIER mois de la fenêtre.

    Paramètres
    ----------
    z : np.ndarray — (T × N) résidus standardisés sur la fenêtre
    a : float      — réactivité aux chocs
    b : float      — persistance
    g : float      — asymétrie (chocs négatifs)

    Retourne
    --------
    R_T : np.ndarray — matrice de corrélation (N × N) au dernier mois
    """
    T, N = z.shape

    Q_bar = np.cov(z.T)
    D = np.sqrt(np.diag(Q_bar))
    Q_bar = Q_bar / np.outer(D, D)

    n = z * (z < 0)
    N_bar = (n.T @ n) / T

    Q = Q_bar.copy()

    for t in range(1, T):
        z_prev = z[t - 1].reshape(-1, 1)
        n_prev = n[t - 1].reshape(-1, 1)

        Q = (
            (1 - a - b) * Q_bar
            - g * N_bar
            + a * (z_prev @ z_prev.T)
            + g * (n_prev @ n_prev.T)
            + b * Q
        )

    # Normalisation finale → R_T
    d = np.sqrt(np.diag(Q))
    R_T = Q / np.outer(d, d)

    # Garantir la symétrie numérique
    R_T = (R_T + R_T.T) / 2
    np.fill_diagonal(R_T, 1.0)

    return R_T


# ── Fonction principale (rolling window) ──────────────────────────────────────


def rolling_adcc(
    residuals: pd.DataFrame,
    window_size: int = config_dependance.WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Applique le modèle ADCC avec une fenêtre roulante de 60 mois.

    Même structure que rolling_dcc() avec le paramètre d'asymétrie g.

    Paramètres
    ----------
    residuals   : pd.DataFrame — résidus standardisés z_{i,t} (T x 5)
                                 issus de rolling_garch() dans garch.py
    window_size : int          — taille de la fenêtre roulante (défaut : 60)

    Retourne
    --------
    correlations : pd.DataFrame — matrices R_t sérialisées
                                  index = dates, colonnes = paires de facteurs
    """
    n_obs = len(residuals)
    factors = residuals.columns.tolist()
    N = len(factors)

    pairs = [f"{factors[i]}-{factors[j]}" for i in range(N) for j in range(i + 1, N)]

    results = pd.DataFrame(index=residuals.index, columns=pairs, dtype=float)

    logging.info(
        f"ADCC rolling window : {window_size} mois | {n_obs - window_size} estimations"
    )

    for t in tqdm(range(window_size, n_obs), desc="Rolling ADCC", unit="mois"):

        window = residuals.iloc[t - window_size : t].values  # (60 × 5)

        # 1. Estimation des paramètres (a, b, g)
        a, b, g = _fit_adcc_params(window)

        # 2. Calcul de R_t au dernier mois de la fenêtre
        R_t = _compute_R_t(window, a, b, g)

        # 3. Stocker les corrélations (triangle supérieur)
        date = residuals.index[t]
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                results.at[date, pairs[k]] = R_t[i, j]
                k += 1

    results = results.dropna(how="all")

    logging.info(f"ADCC : {len(results)} matrices R_t calculées")

    return results


# ── Reconstruction de la matrice complète ─────────────────────────────────────


def get_matrix_at(correlations: pd.DataFrame, date, factors: list) -> np.ndarray:
    """
    Reconstruit la matrice de corrélation 5x5 complète à une date donnée.

    Identique à dcc.get_matrix_at() — même interface.
    """
    N = len(factors)
    R = np.eye(N)
    row = correlations.loc[date]

    for i in range(N):
        for j in range(i + 1, N):
            pair = f"{factors[i]}-{factors[j]}"
            rho = float(row[pair])
            R[i, j] = rho
            R[j, i] = rho

    return R


# ── Export / Import ───────────────────────────────────────────────────────────


def save_adcc(
    correlations: pd.DataFrame,
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> Path:
    """Sauvegarde les corrélations ADCC."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        path = output_dir / "correlations_adcc.parquet"
        correlations.to_parquet(path)
    elif fmt == "xlsx":
        path = output_dir / "correlations_adcc.xlsx"
        correlations.to_excel(path)
    else:
        raise ValueError(f"Format non supporté : '{fmt}'.")

    logging.info(f"ADCC corrélations sauvegardées → {path}")
    return path


def load_adcc(
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """Charge les corrélations ADCC depuis un fichier sauvegardé."""
    output_dir = Path(output_dir)

    if fmt == "parquet":
        return pd.read_parquet(output_dir / "correlations_adcc.parquet")
    elif fmt == "xlsx":
        df = pd.read_excel(output_dir / "correlations_adcc.xlsx", index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    else:
        raise ValueError(f"Format non supporté : '{fmt}'.")
