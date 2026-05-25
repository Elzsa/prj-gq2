# src/dependency_structure/dcc.py

"""
Étape 3c — Modèle DCC (Dynamic Conditional Correlation)
Engle (2002)

Input  : résidus standardisés z_{i,t} issus de garch.py
Output : matrices de corrélation dynamiques R_t (5x5) pour chaque mois t

Dynamique DCC :
    Q_t = (1-a-b)*Q̄ + a*z_{t-1}*z'_{t-1} + b*Q_{t-1}
    R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

où :
    Q̄   = corrélation inconditionnelle (moyenne sur la fenêtre)
    a   = réactivité aux chocs récents
    b   = persistance des corrélations passées
    R_t = matrice de corrélation à l'instant t ∈ (0,1)

Référence : Engle, R. (2002). Dynamic conditional correlation.
            Journal of Business & Economic Statistics, 20(3), 339-350.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from config import config_dependance

# ── Log-vraisemblance DCC ─────────────────────────────────────────────────────


def _dcc_loglikelihood(params: np.ndarray, z: np.ndarray) -> float:
    """
    Calcule l'opposé de la log-vraisemblance du modèle DCC(1,1).

    Cette fonction est appelée par l'optimiseur lors de l'estimation des
    paramètres DCC `a` et `b`. Pour un couple donné (a, b), elle reconstruit
    la séquence des matrices de corrélation conditionnelles R_t sur la fenêtre
    considérée, puis évalue la vraisemblance associée aux résidus standardisés.

    Les données d'entrée `z` sont les innovations standardisées issues des
    modèles marginaux AR-GJR-GARCH :

        z_{i,t} = ε_{i,t} / σ_{i,t}

    où :
        - ε_{i,t} est le résidu du facteur i après retrait de la moyenne
          conditionnelle ;
        - σ_{i,t} est la volatilité conditionnelle estimée par GARCH.

    Le modèle DCC fait évoluer la matrice intermédiaire Q_t selon :

        Q_t = (1 - a - b) Q_bar + a z_{t-1} z'_{t-1} + b Q_{t-1}

    où :
        - Q_bar est la corrélation inconditionnelle sur la fenêtre ;
        - z_{t-1} z'_{t-1} est le produit extérieur des innovations
          standardisées passées, qui mesure les co-mouvements récents ;
        - a mesure la réaction aux nouveaux chocs ;
        - b mesure la persistance des corrélations passées.

    La matrice Q_t est ensuite normalisée pour obtenir une vraie matrice de
    corrélation :

        R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)

    La contribution de vraisemblance utilisée est la partie "corrélation" de
    la vraisemblance DCC :

        l_t = -1/2 [ log|R_t| + z_t' R_t^{-1} z_t - z_t' z_t ]

    Le terme `- z_t' z_t` retire la composante déjà prise en compte par les
    modèles GARCH univariés. Il permet donc d'isoler l'apport de la corrélation
    dynamique R_t dans la vraisemblance multivariée.

    Paramètres
    ----------
    params : np.ndarray
        Tableau contenant les deux paramètres DCC [a, b].

    z : np.ndarray
        Matrice de taille (T, N) contenant les résidus standardisés sur la
        fenêtre d'estimation :
            - T : nombre de périodes, par exemple 60 mois ;
            - N : nombre de facteurs, ici 5.

    Retourne
    --------
    neg_ll : float
        Opposé de la log-vraisemblance DCC. Cette quantité est minimisée par
        l'optimiseur, ce qui revient à maximiser la log-vraisemblance.
        Une forte pénalité est renvoyée si une matrice R_t n'est pas définie
        positive.
    """
    a, b = params
    T, N = z.shape

    # Calcul de la covariance empirique
    Q_bar = np.cov(z.T)
    # Normalisation pour obtenir une corrélation (pas une covariance)
    D = np.sqrt(np.diag(Q_bar))
    Q_bar = Q_bar / np.outer(D, D)

    Q = Q_bar.copy()
    # Initialisation de la log-vraisemblance à 0
    ll = 0.0

    for t in range(1, T):
        z_prev = z[t - 1].reshape(-1, 1)  # N × 1

        # Récurrence DCC
        Q = (1 - a - b) * Q_bar + a * (z_prev @ z_prev.T) + b * Q

        # Normalisation → matrice de corrélation R_t
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)

        # Contribution à la log-vraisemblance
        z_t = z[t]
        # Calcul du déterminant de R
        sign, log_det = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10  # R_t non définie positive → pénalité
        R_inv = np.linalg.inv(R)  # Calcul de l'inverse de R_t

        ll += -0.5 * (log_det + z_t @ R_inv @ z_t - z_t @ z_t)

    return -ll  # négatif car on minimise


# ── Estimation des paramètres ─────────────────────────────────────────────────


def _fit_dcc_params(z: np.ndarray) -> tuple[float, float]:
    """
    Estime les paramètres (a, b) du DCC par MLE sur une fenêtre.

    Contrainte : a + b < 1  (stationnarité)
    Bornes     : a ∈ (0, 0.5), b ∈ (0, 0.9999)

    Paramètres
    ----------
    z : np.ndarray — (T x N) résidus standardisés sur la fenêtre

    Retourne
    --------
    (a, b) : tuple[float, float]
    """
    result = minimize(
        fun=_dcc_loglikelihood,
        x0=np.array([0.05, 0.90]),
        args=(z,),
        method="SLSQP",
        bounds=[(1e-6, 0.5), (1e-6, 0.9999)],
        constraints=[{"type": "ineq", "fun": lambda x: 0.9999 - x[0] - x[1]}],
        options={"ftol": 1e-8, "maxiter": 500},
    )
    if result.success:
        return float(result.x[0]), float(result.x[1])
    else:
        logging.warning(
            f"[WARN] DCC : convergence échouée — valeurs initiales utilisées (a=0.05, b=0.90)"
        )
        return 0.05, 0.90


# ── Calcul de R_t sur une fenêtre ─────────────────────────────────────────────


def _compute_R_t(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Calcule la matrice de corrélation R_t au DERNIER mois de la fenêtre.

    C'est R_T (le dernier point de la récurrence) qui sert de prévision
    pour le mois suivant dans l'approche rolling window.

    Paramètres
    ----------
    z : np.ndarray — (T × N) résidus standardisés sur la fenêtre
    a : float      — paramètre de réactivité aux chocs
    b : float      — paramètre de persistance

    Retourne
    --------
    R_T : np.ndarray — matrice de corrélation (N × N) au dernier mois
    """
    T, N = z.shape

    Q_bar = np.cov(z.T)
    D = np.sqrt(np.diag(Q_bar))
    Q_bar = Q_bar / np.outer(D, D)

    Q = Q_bar.copy()

    for t in range(1, T):
        z_prev = z[t - 1].reshape(-1, 1)
        Q = (1 - a - b) * Q_bar + a * (z_prev @ z_prev.T) + b * Q

    # Normalisation finale → R_T
    d = np.sqrt(np.diag(Q))
    R_T = Q / np.outer(d, d)

    # Garantir la symétrie numérique
    R_T = (R_T + R_T.T) / 2
    np.fill_diagonal(R_T, 1.0)

    return R_T


# ── Fonction principale (rolling window) ──────────────────────────────────────


def rolling_dcc(
    residuals: pd.DataFrame,
    window_size: int = config_dependance.WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Applique le modèle DCC avec une fenêtre roulante de 60 mois.

    À chaque mois t :
        1. Estimer (a, b) par MLE sur z[t-60 : t]
        2. Calculer R_t = dernière matrice de corrélation de la récurrence
        3. R_t sert de prévision pour le mois t

    Paramètres
    ----------
    residuals   : pd.DataFrame — résidus standardisés z_{i,t} (T × 5)
                                 issus de rolling_garch() dans garch.py
    window_size : int          — taille de la fenêtre roulante (défaut : 60)

    Retourne
    --------
    correlations : pd.DataFrame — matrices R_t sérialisées
                                  index = dates, colonnes = paires de facteurs
                                  ex: 'MKT_RF-SMB', 'MKT_RF-HML', ...
    """
    n_obs = len(residuals)
    factors = residuals.columns.tolist()
    N = len(factors)

    # Noms des paires (triangle supérieur de la matrice)
    pairs = [f"{factors[i]}-{factors[j]}" for i in range(N) for j in range(i + 1, N)]

    results = pd.DataFrame(index=residuals.index, columns=pairs, dtype=float)

    logging.info(
        f"DCC rolling window : {window_size} mois | {n_obs - window_size} estimations"
    )

    for t in tqdm(range(window_size, n_obs), desc="Rolling DCC", unit="mois"):

        window = residuals.iloc[t - window_size : t].values  # (60 × 5)

        # 1. Estimation des paramètres
        a, b = _fit_dcc_params(window)

        # 2. Calcul de R_t
        R_t = _compute_R_t(window, a, b)

        # 3. Stocker les corrélations (triangle supérieur)
        date = residuals.index[t]
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                results.at[date, pairs[k]] = R_t[i, j]
                k += 1

    results = results.dropna(how="all")

    logging.info(f"DCC : {len(results)} matrices R_t calculées")

    return results


# ── Reconstruction de la matrice complète ─────────────────────────────────────


def get_matrix_at(correlations: pd.DataFrame, date, factors: list) -> np.ndarray:
    """
    Reconstruit la matrice de corrélation 5×5 complète à une date donnée.

    Paramètres
    ----------
    correlations : pd.DataFrame — output de rolling_dcc()
    date         : date cible
    factors      : list         — noms des 5 facteurs (même ordre que residuals)

    Retourne
    --------
    R : np.ndarray — matrice de corrélation (5×5) symétrique
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


def save_dcc(
    correlations: pd.DataFrame,
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> Path:
    """Sauvegarde les corrélations DCC."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        path = output_dir / "correlations_dcc.parquet"
        correlations.to_parquet(path)
    elif fmt == "xlsx":
        path = output_dir / "correlations_dcc.xlsx"
        correlations.to_excel(path)
    else:
        raise ValueError(f"Format non supporté : '{fmt}'.")

    logging.info(f"DCC corrélations sauvegardées → {path}")
    return path


def load_dcc(
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """Charge les corrélations DCC depuis un fichier sauvegardé."""
    output_dir = Path(output_dir)

    if fmt == "parquet":
        return pd.read_parquet(output_dir / "correlations_dcc.parquet")
    elif fmt == "xlsx":
        df = pd.read_excel(output_dir / "correlations_dcc.xlsx", index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    else:
        raise ValueError(f"Format non supporté : '{fmt}'.")
