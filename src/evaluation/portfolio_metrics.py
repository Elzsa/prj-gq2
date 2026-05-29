import numpy as np
import pandas as pd

from src.dependence.data_loading import load_data


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES DE BASE — Tables 8, 9, 10
# ══════════════════════════════════════════════════════════════════════════════


def annualized_return(returns: np.ndarray, freq: int = 12) -> float:
    """Rendement annualisé (données mensuelles → freq=12)."""
    return float(np.mean(returns) * freq * 100)


def annualized_std(returns: np.ndarray, freq: int = 12) -> float:
    return float(np.std(returns, ddof=1) * np.sqrt(freq))


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, freq: int = 12) -> float:
    """Sharpe annualisé = rendement_annuel / vol_annuelle."""
    excess = returns - rf / freq
    if np.std(excess, ddof=1) == 0:
        return np.nan
    return float(np.mean(excess) * freq / (np.std(excess, ddof=1) * np.sqrt(freq)))


def sortino_ratio(returns: np.ndarray, rf: float = 0.0, freq: int = 12) -> float:
    """Sortino annualisé = rendement_annuel / vol_downside_annuelle."""
    excess = returns - rf / freq
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside, ddof=1) == 0:
        return np.nan
    downside_std = np.std(downside, ddof=1) * np.sqrt(freq)
    return float(np.mean(excess) * freq / downside_std)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum Drawdown en % (pic → creux de la valeur cumulée), valeur positive."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(-np.min(drawdowns) * 100)


def cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    CVaR (Expected Shortfall) à un niveau de confiance donné.

    Retourne une valeur positive (perte espérée au-delà du VaR).
    Ex : confidence_level=0.95 → CVaR à 95%, soit la moyenne des 5% pires pertes.
    """
    q = 1 - confidence_level
    var_threshold = np.quantile(returns, q)
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        return np.nan
    return float(-np.mean(tail))


def return_over_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    freq: int = 12,
) -> float:
    """Return/CVaR annualisé — ratio utilisé dans les Tables 9 et 10."""
    ann_ret = annualized_return(returns, freq) / 100
    cvar_val = cvar(returns, confidence_level)
    if cvar_val == 0 or np.isnan(cvar_val):
        return np.nan
    return float(ann_ret / cvar_val)


# ══════════════════════════════════════════════════════════════════════════════
# CDB — Appendix E (Christoffersen et al. 2012)
# ══════════════════════════════════════════════════════════════════════════════


def cdb(
    portfolio_returns: np.ndarray,
    weights: np.ndarray,
    individual_returns: np.ndarray,
    q: float = 0.01,
) -> float:
    """
    Conditional Diversification Benefit (Appendix E, eq. E.4).

    CDB = (CVaR_bar - CVaR_portfolio) / (CVaR_bar - CVaR_underbar)

    où :
        CVaR_bar       = Σ_i w_i * CVaR_i   (borne sup : pas de diversification)
        CVaR_underbar  = -F⁻¹_{portfolio}(q) (borne inf : VaR du portefeuille)
        CVaR_portfolio = CVaR réel du portefeuille

    Le papier utilise q=0.01 (99% confidence) pour le CDB (footnote 12).

    Paramètres
    ----------
    portfolio_returns   : (T,) rendements du portefeuille
    weights             : (N,) poids moyens sur la période OOS
    individual_returns  : (T, N) rendements des N facteurs
    q                   : quantile de queue (défaut 0.01 = 99% confidence)

    Retourne
    --------
    CDB ∈ [0, 1]
    """
    # CVaR individuel de chaque facteur
    individual_cvars = np.array([
        cvar(individual_returns[:, i], confidence_level=1 - q)
        for i in range(individual_returns.shape[1])
    ])

    # Borne supérieure : somme pondérée des CVaRs individuels (pas de diversification)
    cvar_bar = float(weights @ individual_cvars)

    # CVaR réel du portefeuille
    cvar_portfolio = cvar(portfolio_returns, confidence_level=1 - q)

    # Borne inférieure : VaR du portefeuille (perte parfaitement diversifiée)
    cvar_underbar = float(-np.quantile(portfolio_returns, q))

    denom = cvar_bar - cvar_underbar
    if denom == 0 or np.isnan(denom):
        return np.nan

    return float((cvar_bar - cvar_portfolio) / denom)


# ══════════════════════════════════════════════════════════════════════════════
# AGRÉGATION — Tables 8, 9, 10
# ══════════════════════════════════════════════════════════════════════════════


def compute_portfolio_metrics(
    portfolio_returns: np.ndarray,
    weights: np.ndarray,
    individual_returns: np.ndarray,
    confidence_level: float = 0.95,
    freq: int = 12,
) -> dict:
    """
    Calcule toutes les métriques pour un portefeuille.

    Paramètres
    ----------
    portfolio_returns  : (T,) rendements mensuels du portefeuille OOS
    weights            : (N,) poids moyens sur la période OOS
    individual_returns : (T, N) rendements des N facteurs
    confidence_level   : niveau CVaR pour Return/CVaR (0.95 ou 0.99)
    freq               : fréquence annualisation (12 pour mensuel)

    Retourne
    --------
    dict avec clés : Annualized Return, Sharpe, Sortino, MDD, Return/CVaR, CDB
    """
    return {
        "Annualized Return (%)": annualized_return(portfolio_returns, freq),
        "Sharpe": sharpe_ratio(portfolio_returns, freq=freq),
        "Sortino": sortino_ratio(portfolio_returns, freq=freq),
        "MDD (%)": max_drawdown(portfolio_returns),
        "Return/CVaR": return_over_cvar(portfolio_returns, confidence_level, freq),
        "CDB": cdb(portfolio_returns, weights, individual_returns),
    }


def build_table_8_to_10(
    results_dict: dict[str, dict],
) -> pd.DataFrame:
    """
    Construit les Tables 8, 9 ou 10 du papier.

    Paramètres
    ----------
    results_dict : {nom_stratégie: compute_portfolio_metrics(...)}
                   Les noms doivent suivre le format "{FORECAST}-{DEP}-SKT"
                   ex. {"RW-DCC-SKT": {...}, "DMA-GAS-SKT": {...}, ...}

    Retourne
    --------
    DataFrame avec lignes = stratégies + lignes de moyennes,
    colonnes = métriques (format papier, sans Sharpe).
    """
    metrics_cols = ["Annualized Return (%)", "Return/CVaR", "Sortino", "MDD (%)", "CDB"]
    rename = {"Sortino": "Sortino ratio", "Annualized Return (%)": "Annualized return (%)"}

    df = pd.DataFrame(results_dict).T[metrics_cols]

    # Déduire le groupe forecasting et le modèle de dépendance depuis le nom
    def _forecast(name: str) -> str:
        return name.split("-")[0]

    def _dep(name: str) -> str:
        parts = name.split("-")
        # SC-SVR a un tiret dans son nom → prendre l'avant-dernier composant
        return parts[-2] if len(parts) >= 3 else parts[1]

    strategies = list(results_dict.keys())
    forecast_groups = dict.fromkeys(_forecast(s) for s in strategies)  # ordre insertion

    rows = []
    for fg in forecast_groups:
        group_strats = [s for s in strategies if _forecast(s) == fg]
        for s in group_strats:
            rows.append(df.loc[s])
        # ligne Average par groupe forecasting
        avg_row = df.loc[group_strats].mean()
        avg_row.name = "Average"
        rows.append(avg_row)

    # Total Average
    total_avg = df.mean()
    total_avg.name = "Total Average"
    rows.append(total_avg)

    # Moyennes par modèle de dépendance (DCC, ADCC, GAS)
    dep_groups = dict.fromkeys(_dep(s) for s in strategies)
    for dg in dep_groups:
        dep_strats = [s for s in strategies if _dep(s) == dg]
        dep_avg = df.loc[dep_strats].mean()
        dep_avg.name = f"{dg} Average"
        rows.append(dep_avg)

    result = pd.DataFrame(rows)
    result = result.rename(columns=rename)
    return result



def build_table_9_panel_a() -> pd.DataFrame:
    """
    Reproduit la Table 9 Panel A :
    performances des 5 facteurs + portefeuille 1/N sur 2000-01 à 2017-08.
    """
    factors_df, _, _ = load_data()
    factors = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]

    # période out-of-sample du papier
    oos = factors_df.loc["2000-01-01":"2017-08-01", factors].copy()

    results = {}

    # Facteurs seuls
    for i, factor in enumerate(factors):
        r = oos[factor].dropna().values

        results[factor] = {
            "Annualized return (%)": annualized_return(r),
            "Return/CVaR": return_over_cvar(r, confidence_level=0.95),
            "Sortino ratio": sortino_ratio(r),
            "MDD (%)": max_drawdown(r),
            "CDB": np.nan,
        }

    # Portefeuille 1/N
    weights_1n = np.ones(len(factors)) / len(factors)
    r_1n = oos.values @ weights_1n

    results["1/N"] = {
        "Annualized return (%)": annualized_return(r_1n),
        "Return/CVaR": return_over_cvar(r_1n, confidence_level=0.95),
        "Sortino ratio": sortino_ratio(r_1n),
        "MDD (%)": max_drawdown(r_1n),
        "CDB": cdb(
            portfolio_returns=r_1n,
            weights=weights_1n,
            individual_returns=oos.values,
            q=0.01,
        ),
    }

    return pd.DataFrame(results).T
