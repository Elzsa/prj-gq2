# src/dependency_structure/garch.py

"""
Étape 3a & 3b — Modélisation marginale : AR(p) + GJR-GARCH(p,1,q) + skewed-t
               + Transformation PIT

Pour chaque facteur i et chaque mois t (fenêtre roulante de 60 mois) :
    1. Sélection des ordres (ar_p, garch_p, garch_q) par critère BIC
    2. Estimation du modèle AR(ar_p) + GJR-GARCH(garch_p, 1, garch_q)
       avec distribution skewed-t de Hansen (1994)
    3. Calcul des résidus standardisés : z_{i,t} = ε_{i,t} / σ_{i,t}
    4. Transformation PIT : u_{i,t} = F_skt(z_{i,t} ; η_i, λ_i) ∈ [0, 1]

Outputs :
    - residuals : DataFrame des résidus standardisés z_{i,t}
    - uniforms  : DataFrame des pseudo-uniformes u_{i,t}
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from tqdm import tqdm

from config import config_dependance
from src.dependence.pit import pit_transform

# ── Fonctions utilitaires ─────────────────────────────────────────────────────


def _select_ar_garch_order(
    series: pd.Series,
    max_ar_lags: int | None = None,
    max_garch_p: int | None = None,
    max_garch_q: int | None = None,
) -> tuple[int, int, int]:
    """
    Sélectionne les ordres optimaux (ar_p, garch_p, garch_q) par BIC.

    Les ordres sont sélectionnés à CHAQUE fenêtre roulante — cohérent avec
    l'esprit du rolling window : les dynamiques changent dans le temps.
    Pour accélérer, fixer MAX_GARCH_P = MAX_GARCH_Q = MAX_AR_LAGS = 1
    dans config.py revient à des ordres fixes sans sélection BIC.

    Paramètres
    ----------
    series      : pd.Series — rendements sur la fenêtre courante
    max_ar_lags : int       — ordre AR maximum à tester (défaut : config.MAX_AR_LAGS)
    max_garch_p : int       — ordre ARCH maximum à tester (défaut : config.MAX_GARCH_P)
    max_garch_q : int       — ordre GARCH maximum à tester (défaut : config.MAX_GARCH_Q)

    Retourne
    --------
    (best_ar_p, best_garch_p, best_garch_q) : tuple[int, int, int]
    """
    # Lecture de config au moment de l'appel
    if max_ar_lags is None:
        max_ar_lags = config_dependance.MAX_AR_LAGS
    if max_garch_p is None:
        max_garch_p = config_dependance.MAX_GARCH_P
    if max_garch_q is None:
        max_garch_q = config_dependance.MAX_GARCH_Q

    best_bic = np.inf
    best_orders = (1, 1, 1)  # valeur par défaut si tout échoue

    for ar_p in range(1, max_ar_lags + 1):
        for garch_p in range(1, max_garch_p + 1):
            for garch_q in range(1, max_garch_q + 1):
                try:
                    model = arch_model(
                        series,
                        mean="AR",
                        lags=ar_p,
                        vol="GARCH",
                        p=garch_p,
                        o=config_dependance.MAX_GARCH_O,  # fixe = 1 → active l'asymétrie GJR
                        q=garch_q,
                        dist="skewt",
                        rescale=False,
                    )
                    result = model.fit(disp="off", show_warning=False)
                    if result.bic < best_bic:
                        best_bic = result.bic
                        best_orders = (ar_p, garch_p, garch_q)
                except Exception as e:
                    logging.warning(
                        f"[WARN] AR({ar_p})-GJR-GARCH({garch_p},1,{garch_q}) "
                        f"a échoué lors de la sélection BIC : {e}"
                    )
                    continue

    return best_orders


def _fit_single_window(series: pd.Series) -> tuple[float, float] | None:
    """
    Estime AR(ar_p) + GJR-GARCH(garch_p, 1, garch_q) sur une fenêtre de 60 mois
    et retourne le résidu standardisé ET le pseudo-uniforme du dernier mois.

    Étapes :
        1. Sélection des ordres par BIC
        2. Estimation du modèle complet
        3. z_t = ε_t / σ_t  (résidu standardisé du dernier mois)
        4. u_t = F_skt(z_t ; η, λ)  (transformation PIT)

    Paramètres
    ----------
    series : pd.Series — rendements sur la fenêtre (60 observations)

    Retourne
    --------
    (z_t, u_t) : tuple[float, float] — résidu standardisé et pseudo-uniforme
                                        du dernier mois, ou None si échec
    """
    try:
        # 1. Sélection des ordres par BIC
        ar_p, garch_p, garch_q = _select_ar_garch_order(series)

        # 2. Estimation du modèle complet
        # NOTE : o=config.MAX_GARCH_O est fixe (= 1) — choix de spécification,
        # pas soumis à la sélection BIC. Active le terme asymétrique GJR :
        # γ ε²_{t-1} 1[ε_{t-1} < 0]  (effet de levier)
        model = arch_model(
            series,
            mean="AR",
            lags=ar_p,
            vol="GARCH",
            p=garch_p,
            o=config_dependance.MAX_GARCH_O,
            q=garch_q,
            dist="skewt",
            rescale=False,
        )
        result = model.fit(disp="off", show_warning=False)

        # 3. Résidu standardisé du dernier mois : z_t = ε_t / σ_t
        # NOTE : result.std_resid est calculé automatiquement par arch
        # Vérification : result.resid.iloc[-1] / result.conditional_volatility.iloc[-1]
        z_t = float(result.std_resid.iloc[-1])

        # 4. Paramètres de la skewed-t estimés par arch
        eta = float(result.params["eta"])  # degrés de liberté (η > 2)
        lam = float(result.params["lambda"])  # asymétrie (-1 < λ < 1)

        # 5. Transformation PIT : u_t = F_skt(z_t ; η, λ) ∈ (0, 1)
        u_t = pit_transform(z_t, eta, lam)

        return z_t, u_t

    except Exception as e:
        logging.warning(f"[WARN] Estimation échouée : {e}")
        return None


# ── Fonction principale ───────────────────────────────────────────────────────


def rolling_garch(
    factors: pd.DataFrame,
    window_size: int = config_dependance.WINDOW_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applique AR(ar_p) + GJR-GARCH(garch_p, 1, garch_q) + PIT avec une fenêtre
    roulante sur les 5 facteurs.

    À chaque mois t, on estime le modèle sur les [t - window_size : t] observations
    et on récupère :
        - z_{i,t} = résidu standardisé  (pour diagnostic)
        - u_{i,t} = pseudo-uniforme PIT (input de la copule)

    Paramètres
    ----------
    factors     : pd.DataFrame — rendements des 5 facteurs (toutes périodes)
    window_size : int          — taille de la fenêtre roulante (défaut : config.WINDOW_SIZE)

    Retourne
    --------
    residuals : pd.DataFrame — résidus standardisés z_{i,t}
    uniforms  : pd.DataFrame — pseudo-uniformes u_{i,t} ∈ (0, 1)
                               Les window_size premiers mois sont absents
                               (pas d'historique suffisant)
    """
    n_obs = len(factors)
    residuals = pd.DataFrame(index=factors.index, columns=factors.columns, dtype=float)
    uniforms = pd.DataFrame(index=factors.index, columns=factors.columns, dtype=float)

    logging.info(
        f"Fenêtre roulante : {window_size} mois | {n_obs - window_size} estimations par facteur"
    )

    for t in tqdm(range(window_size, n_obs), desc="Rolling GARCH", unit="mois"):
        window = factors.iloc[t - window_size + 1 : t + 1]
        # Ancienne version : window = factors.iloc[t - window_size : t]

        # Estimation pour chaque facteur indépendamment
        for col in factors.columns:
            output = _fit_single_window(window[col])
            if output is not None:
                z_t, u_t = output
                residuals.at[factors.index[t], col] = z_t
                uniforms.at[factors.index[t], col] = u_t

    # Supprimer les lignes sans résidu (les window_size premiers mois)
    residuals = residuals.dropna(how="all")
    uniforms = uniforms.dropna(how="all")

    logging.info(
        f"Résidus calculés : {len(residuals)} observations × {len(factors.columns)} facteurs"
    )

    return residuals, uniforms


# ── Export / Import des résultats ─────────────────────────────────────────────


def save_residuals(
    residuals: pd.DataFrame,
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> Path:
    """
    Sauvegarde les résidus standardisés dans un fichier parquet ou xlsx.

    Paramètres
    ----------
    residuals  : pd.DataFrame — résidus standardisés issus de rolling_garch()
    output_dir : Path         — dossier de destination (défaut : config.RESULTS_DIR)
    fmt        : str          — format de sortie : 'parquet' (défaut) ou 'xlsx'

    Retourne
    --------
    output_path : Path — chemin du fichier créé
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        output_path = output_dir / "residuals_garch.parquet"
        residuals.to_parquet(output_path)
    elif fmt == "xlsx":
        output_path = output_dir / "residuals_garch.xlsx"
        residuals.to_excel(output_path)
    else:
        raise ValueError(f"Format non supporté : '{fmt}'. Choisir 'parquet' ou 'xlsx'.")

    logging.info(f"Résidus sauvegardés → {output_path}")
    return output_path


def save_uniforms(
    uniforms: pd.DataFrame,
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> Path:
    """
    Sauvegarde les pseudo-uniformes PIT dans un fichier parquet ou xlsx.

    Paramètres
    ----------
    uniforms   : pd.DataFrame — pseudo-uniformes issus de rolling_garch()
    output_dir : Path         — dossier de destination (défaut : config.RESULTS_DIR)
    fmt        : str          — format de sortie : 'parquet' (défaut) ou 'xlsx'

    Retourne
    --------
    output_path : Path — chemin du fichier créé
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        output_path = output_dir / "uniforms_pit.parquet"
        uniforms.to_parquet(output_path)
    elif fmt == "xlsx":
        output_path = output_dir / "uniforms_pit.xlsx"
        uniforms.to_excel(output_path)
    else:
        raise ValueError(f"Format non supporté : '{fmt}'. Choisir 'parquet' ou 'xlsx'.")

    logging.info(f"Uniformes sauvegardés → {output_path}")
    return output_path


def load_residuals(
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """
    Charge les résidus standardisés depuis un fichier sauvegardé.

    Paramètres
    ----------
    output_dir : Path — dossier source (défaut : config.RESULTS_DIR)
    fmt        : str  — format : 'parquet' (défaut) ou 'xlsx'

    Retourne
    --------
    residuals : pd.DataFrame
    """
    output_dir = Path(output_dir)

    if fmt == "parquet":
        path = output_dir / "residuals_garch.parquet"
        return pd.read_parquet(path)
    elif fmt == "xlsx":
        path = output_dir / "residuals_garch.xlsx"
        df = pd.read_excel(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    else:
        raise ValueError(f"Format non supporté : '{fmt}'. Choisir 'parquet' ou 'xlsx'.")


def load_uniforms(
    output_dir: Path = config_dependance.RESULTS_DIR,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """
    Charge les pseudo-uniformes PIT depuis un fichier sauvegardé.

    Paramètres
    ----------
    output_dir : Path — dossier source (défaut : config.RESULTS_DIR)
    fmt        : str  — format : 'parquet' (défaut) ou 'xlsx'

    Retourne
    --------
    uniforms : pd.DataFrame
    """
    output_dir = Path(output_dir)

    if fmt == "parquet":
        path = output_dir / "uniforms_pit.parquet"
        return pd.read_parquet(path)
    elif fmt == "xlsx":
        path = output_dir / "uniforms_pit.xlsx"
        df = pd.read_excel(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    else:
        raise ValueError(f"Format non supporté : '{fmt}'. Choisir 'parquet' ou 'xlsx'.")
