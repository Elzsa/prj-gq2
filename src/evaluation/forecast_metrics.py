import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES DE BASE — Table 4 : MAE, MAPE, RMSE, Theil-U
# ══════════════════════════════════════════════════════════════════════════════


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Division par y_true peut exploser si retours proches de 0 — comportement attendu (cf. Table 4 : valeurs > 100%)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Theil U1 : RMSE(ŷ, y) / (RMS(ŷ) + RMS(y))"""
    rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rms_pred = np.sqrt(np.mean(y_pred**2))
    rms_true = np.sqrt(np.mean(y_true**2))
    return float(rmse_val / (rms_pred + rms_true))


# ══════════════════════════════════════════════════════════════════════════════
# TESTS STATISTIQUES — Table 5 : PT et DM
# ══════════════════════════════════════════════════════════════════════════════


def pt_test(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Test de Pesaran-Timmermann (1992) — précision directionnelle.

    H0 : les prévisions sont indépendantes des vraies valeurs (pas de signal directionnel).
    Statistique asymptotiquement N(0,1) sous H0.

    Paramètres
    ----------
    y_true : rendements réalisés
    y_pred : prévisions du modèle

    Retourne
    --------
    (stat, p_value) : stat > 0 = meilleur que le hasard
    """
    n = len(y_true)
    p_hat = np.mean(np.sign(y_true) == np.sign(y_pred))
    p_y = np.mean(y_true > 0)
    p_f = np.mean(y_pred > 0)
    p_star = p_y * p_f + (1 - p_y) * (1 - p_f)

    var_p = p_star * (1 - p_star) / n
    var_pstar = (
        (2 * p_y - 1) ** 2 * p_f * (1 - p_f)
        + (2 * p_f - 1) ** 2 * p_y * (1 - p_y)
    ) / n

    var_diff = var_p - var_pstar
    if var_diff <= 0:
        return np.nan, np.nan

    stat = (p_hat - p_star) / np.sqrt(var_diff)
    pval = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(pval)


def dm_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss: str = "mse",
) -> tuple[float, float]:
    """
    Test de Diebold-Mariano (1995) — égalité de précision entre deux modèles.

    H0 : E[L(e1_t) - L(e2_t)] = 0
    DM < 0 → y_pred2 plus précis que y_pred1 (dans le papier : y_pred2 = DMA).

    Paramètres
    ----------
    y_true  : rendements réalisés
    y_pred1 : prévisions du modèle à tester (ex. RW, SVR, SC-SVR)
    y_pred2 : prévisions du modèle de référence (ex. DMA)
    loss    : 'mse' (défaut, utilisé dans le papier) ou 'mae'

    Retourne
    --------
    (stat, p_value)
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    if loss == "mse":
        d = e1**2 - e2**2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Loss non supportée : '{loss}'. Choisir 'mse' ou 'mae'.")

    n = len(d)
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1) / n

    if var_d <= 0:
        return np.nan, np.nan

    stat = d_bar / np.sqrt(var_d)
    pval = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(pval)


# ══════════════════════════════════════════════════════════════════════════════
# AGRÉGATION — Tables 4 et 5
# ══════════════════════════════════════════════════════════════════════════════


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_benchmark: np.ndarray | None = None,
) -> dict:
    """
    Calcule toutes les métriques pour un modèle sur un facteur.

    Paramètres
    ----------
    y_true           : rendements réalisés OOS (212 observations)
    y_pred           : prévisions du modèle
    y_pred_benchmark : prévisions du modèle de référence pour le DM (ex. DMA)
                       Si None, le DM n'est pas calculé.

    Retourne
    --------
    dict avec clés : MAE, MAPE, RMSE, Theil-U, PT, PT_pval, [DM, DM_pval]
    """
    pt_stat, pt_pval = pt_test(y_true, y_pred)

    metrics = {
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "Theil-U": theil_u(y_true, y_pred),
        "PT": pt_stat,
        "PT_pval": pt_pval,
    }

    if y_pred_benchmark is not None:
        dm_stat, dm_pval = dm_test(y_true, y_pred, y_pred_benchmark)
        metrics["DM"] = dm_stat
        metrics["DM_pval"] = dm_pval

    return metrics


def build_table_4(
    y_true_dict: dict[str, np.ndarray],
    predictions_dict: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """
    Construit la Table 4 du papier (performances OOS par facteur et modèle).

    Paramètres
    ----------
    y_true_dict      : {facteur: y_true}  ex. {"MKT": array, "SMB": array, ...}
    predictions_dict : {modèle: {facteur: y_pred}}
                       ex. {"RW": {"MKT": array, ...}, "DMA": {"MKT": array, ...}}

    Retourne
    --------
    DataFrame multi-index (facteur, métrique) × modèle
    """
    rows = []
    metrics_names = ["MAE", "MAPE", "RMSE", "Theil-U"]

    for factor, y_true in y_true_dict.items():
        for metric in metrics_names:
            row = {"Factor": factor, "Statistic": metric}
            for model, preds in predictions_dict.items():
                y_pred = preds[factor]
                row[model] = compute_metrics(y_true, y_pred)[metric]
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["Factor", "Statistic"])
    return df


def build_table_5(
    y_true_dict: dict[str, np.ndarray],
    predictions_dict: dict[str, dict[str, np.ndarray]],
    benchmark_model: str = "DMA",
) -> pd.DataFrame:
    """
    Construit la Table 5 du papier (PT et DM par facteur et modèle).

    Paramètres
    ----------
    y_true_dict      : {facteur: y_true}
    predictions_dict : {modèle: {facteur: y_pred}}
    benchmark_model  : modèle de référence pour le DM (défaut : 'DMA')

    Retourne
    --------
    DataFrame multi-index (statistique, facteur) × modèle
    """
    rows = []

    for factor, y_true in y_true_dict.items():
        y_bench = predictions_dict[benchmark_model][factor]

        for model, preds in predictions_dict.items():
            y_pred = preds[factor]
            pt_stat, _ = pt_test(y_true, y_pred)

            if model == benchmark_model:
                dm_stat = np.nan
            else:
                dm_stat, _ = dm_test(y_true, y_pred, y_bench)

            rows.append({
                "Factor": factor,
                "Model": model,
                "PT": pt_stat,
                "DM": dm_stat,
            })

    df = pd.DataFrame(rows).set_index(["Factor", "Model"])
    return df
