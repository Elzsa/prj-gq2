#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
# src/04_portfolio/mean_cvar_optimization_.py

"""
Section 5.2 — Optimisation Moyenne-CVaR par copule skewed-t
Réplication fidèle de la Section 5.2 du papier :
    Zhao, Stasinakis, Sermpinis & Da Silva Fernandes (2019)
    Int J Fin Econ, 24, 1443-1463.

Pipeline :
    1. Chargement des données (rendements, prévisions, corrélations, PIT, résidus)
    2. Estimation des paramètres marginaux Hansen skewed-t (Appendice B)
    3. Volatilités GARCH(1,1) rolling avec cache disque
    4. Estimation roulante des paramètres de copule GH skewed-t (60 mois, Appendice D)
    5. Boucle principale — structure optimisée :
         Pour chaque (corr_model, date) :
           a. Simuler Q rendements avec mu=0 (copule + GARCH) — UNE SEULE FOIS
           b. Pour chaque prev_model : r_sim = r_zero + mu_t
           c. Maximiser directement (return − rf) / CVaR_β via SLSQP (Section 5.2)
    6. Deux niveaux : β = 0.95 et β = 0.99
    7. Deux cadres : long-only (Panel B) et 130/30 (Panel C)
    8. Benchmarks : 1/N et RW
    9. Métriques : rendement annualisé, return/CVaR, Sortino, MDD, CDB
    10. Sauvegarde CSV

Combinaisons (Tables 9 et 10 du papier) :
    Prévisions × Corrélation = {RW, SVR, SC-SVR, DMA} × {DCC, ADCC, GAS}

Remarque sur le prédicteur "Best" :
    best_individual.csv ne couvre que l'IS (1965-1999) → absent des outputs OOS.
    Les combinaisons "Best-XXX-SKT" ne sont pas calculées.

Stratégie d'optimisation :
    Le portefeuille tangent maximise directement (μ_p − rf) / CVaR_β (eq. 15 du
    papier, Section 5.2). La CVaR est estimée empiriquement sur les Q rendements
    simulés via la copule GH skewed-t. La maximisation utilise SLSQP avec
    plusieurs points de départ, ce qui est plus rapide que le sweep de frontière
    et converge vers le même optimum global (return/CVaR est quasi-concave).

Performance attendue :
    GARCH rolling (1× si cache absent) : ~20-30 min
    Copule roulante                     : ~5-10 min
    Optimisations CVaR (Q=5000)        : ~15-25 min
    Total                               : ~45-65 min
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from arch import arch_model
from tqdm import tqdm

from config.splits import OOS_START, OOS_END

# Import dynamique de monte_carlo.py (dossier "04_portfolio" non-importable directement)
import importlib.util as _iutil
_mc_path = Path(__file__).resolve().parent / "monte_carlo.py"
_spec    = _iutil.spec_from_file_location("monte_carlo", _mc_path)
_mc_mod  = _iutil.module_from_spec(_spec)
_spec.loader.exec_module(_mc_mod)

fit_hansen_skt             = _mc_mod.fit_hansen_skt
estimate_copula_params     = _mc_mod.estimate_copula_params
simulate_portfolio_returns = _mc_mod.simulate_portfolio_returns
corr_row_to_matrix         = _mc_mod.corr_row_to_matrix
FACTEURS                   = _mc_mod.FACTEURS
N_FACTORS                  = _mc_mod.N_FACTORS

# ──────────────────────────────────────────────────────────────────────────────
# CHEMINS ET CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]

CHEMIN_RETURNS = ROOT / "data" / "monthly_log_returns.csv"
CHEMIN_FF_RAW  = ROOT / "data" / "F-F_Research_Data_5_Factors_2x3.csv"
CHEMIN_DMA     = ROOT / "data" / "02_forecasting" / "previsions_dma.csv"
CHEMIN_SC_SVR  = ROOT / "data" / "02_forecasting" / "previsions_sc_svr.csv"
CHEMIN_SVR     = ROOT / "data" / "02_forecasting" / "previsions_svr.csv"
CHEMIN_DCC     = ROOT / "results" / "ext" / "correlations_dcc.parquet"
CHEMIN_ADCC    = ROOT / "results" / "ext" / "correlations_adcc.parquet"
CHEMIN_GAS     = ROOT / "results" / "ext" / "correlations_gas.parquet"
CHEMIN_PIT     = ROOT / "results" / "ext" / "uniforms_pit.parquet"
CHEMIN_RESID   = ROOT / "results" / "ext" / "residuals_garch.parquet"

DIR_RESULTS      = ROOT / "results" / "optimization"
CACHE_GARCH_VOLS = DIR_RESULTS / "garch_vols_cache.csv"
DIR_RESULTS_CVAR = ROOT / "results" / "optimization_CVaR"
DIR_RESULTS_95   = DIR_RESULTS_CVAR / "CVaR_95"
DIR_RESULTS_99   = DIR_RESULTS_CVAR / "CVaR_99"
DIR_RESULTS_BEST = DIR_RESULTS_CVAR / "best_portfolio_tangent"

FACTEURS      = ["MKT", "SMB", "HML", "RMW", "CMA"]
FACTEURS_CORR = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]

WINDOW_COPULA = 60    # fenêtre roulante copule (mois), Section 5.2
WINDOW_GARCH  = 60    # fenêtre GARCH rolling (mois)

# Leviers 130/30 (papier, note 11)
LONG_TARGET  = 1.30
SHORT_TARGET = 0.30

# Simulations Monte Carlo (Q = nombre de paths)
# Q=5000 : bon compromis précision/vitesse avec SLSQP
N_SIMS = 5_000

# Niveaux de confiance CVaR
BETAS = [0.95, 0.99]

# Graine globale pour reproductibilité
GLOBAL_SEED = 42

# Réglages SLSQP. Les warm-starts entre dates permettent de réduire les itérations.
SLSQP_FTOL          = 1e-8
SLSQP_MAXITER_LO    = 120
SLSQP_MAXITER_13030 = 180


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 : CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────────

def _charger_rendements() -> pd.DataFrame:
    df = pd.read_csv(
        CHEMIN_RETURNS, index_col=0, parse_dates=True, date_format="%Y-%m-%d"
    )
    return df[FACTEURS]


def _charger_rf() -> pd.Series:
    ff = pd.read_csv(CHEMIN_FF_RAW, skiprows=4, index_col=0, dtype=str)
    ff = ff[ff.index.str.match(r"^\d{6}$", na=False)].copy()
    rf = ff[["RF"]].apply(pd.to_numeric, errors="coerce").dropna()
    rf.index = pd.to_datetime(rf.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    rf.index.name = "date"
    return np.log(1.0 + rf["RF"] / 100.0)


def _charger_previsions(chemin: Path) -> pd.DataFrame:
    df = pd.read_csv(chemin, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    return df[FACTEURS]


def _charger_previsions_rw(monthly: pd.DataFrame, oos_dates: pd.DatetimeIndex) -> pd.DataFrame:
    return monthly.shift(1).reindex(oos_dates)[FACTEURS]


def _align_corr_to_eom(corr: pd.DataFrame) -> pd.DataFrame:
    df = corr.copy()
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
    return df


def _charger_pit() -> pd.DataFrame:
    pit = pd.read_parquet(CHEMIN_PIT)
    pit.index = pd.to_datetime(pit.index) + pd.offsets.MonthEnd(0)
    return pit[FACTEURS_CORR].rename(columns={"MKT_RF": "MKT"})


def _charger_residus() -> pd.DataFrame:
    resid = pd.read_parquet(CHEMIN_RESID)
    resid.index = pd.to_datetime(resid.index) + pd.offsets.MonthEnd(0)
    return resid[FACTEURS_CORR].rename(columns={"MKT_RF": "MKT"})


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 : VOLATILITÉS GARCH(1,1) ROLLING AVEC CACHE
# ──────────────────────────────────────────────────────────────────────────────

def _compute_garch_vol_rolling(
    monthly: pd.DataFrame,
    oos_dates: pd.DatetimeIndex,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Volatilité conditionnelle GARCH(1,1) rolling (Appendice B).
    Fenêtre de 60 mois. Cache disque dans results/optimization/garch_vols_cache.csv.
    """
    # Charger le cache si disponible et complet
    if CACHE_GARCH_VOLS.exists():
        cache = pd.read_csv(CACHE_GARCH_VOLS, index_col=0, parse_dates=True)
        if all(f in cache.columns for f in FACTEURS):
            cached_dates = cache.index.intersection(oos_dates)
            if len(cached_dates) == len(oos_dates):
                if verbose:
                    print(f"  Cache GARCH chargé : {CACHE_GARCH_VOLS.name}")
                return cache.reindex(oos_dates).astype(float)

    if verbose:
        print("  (aucun cache valide — calcul complet...)")

    vols = pd.DataFrame(index=oos_dates, columns=FACTEURS, dtype=float)

    for facteur in FACTEURS:
        serie = monthly[facteur]
        for date_t in tqdm(oos_dates, desc=f"  GARCH {facteur}", leave=False):
            try:
                idx_t = serie.index.get_loc(date_t)
            except KeyError:
                continue
            start  = max(0, idx_t - WINDOW_GARCH + 1)
            window = serie.iloc[start : idx_t + 1]
            if len(window) < 10:
                vols.at[date_t, facteur] = float(window.std()) if len(window) > 1 else np.nan
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mdl = arch_model(window * 100.0, mean="Zero", vol="GARCH",
                                     p=1, q=1, dist="normal", rescale=False)
                    fit = mdl.fit(disp="off", show_warning=False)
                    vols.at[date_t, facteur] = float(fit.conditional_volatility.iloc[-1]) / 100.0
            except Exception:
                vols.at[date_t, facteur] = float(window.std())

    # Sauvegarder le cache
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    vols.to_csv(CACHE_GARCH_VOLS, date_format="%Y-%m-%d")
    if verbose:
        print(f"  Cache sauvegardé : {CACHE_GARCH_VOLS.name}")
    return vols.astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 : PARAMÈTRES MARGINAUX HANSEN (1994)
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_marginal_params(residus: pd.DataFrame) -> list[tuple[float, float]]:
    """Estime (η_i, λ_i) de Hansen pour chaque facteur via MLE."""
    params = []
    for f in FACTEURS:
        if f not in residus.columns:
            params.append((8.0, 0.0))
            continue
        z_vals = residus[f].dropna().values
        nu_hat, lam_hat = fit_hansen_skt(z_vals)
        params.append((nu_hat, lam_hat))
    return params


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 : ESTIMATION ROULANTE DES PARAMÈTRES DE COPULE
# ──────────────────────────────────────────────────────────────────────────────

def _build_R_window(corr_df: pd.DataFrame, dates_window: pd.DatetimeIndex) -> list[np.ndarray]:
    R_list = []
    for d in dates_window:
        if d in corr_df.index:
            R_list.append(corr_row_to_matrix(corr_df.loc[d], FACTEURS_CORR))
        else:
            R_list.append(np.eye(N_FACTORS))
    return R_list


def _estimate_copula_rolling(
    pit_df: pd.DataFrame,
    corr_df_eom: pd.DataFrame,
    oos_dates: pd.DatetimeIndex,
    label: str = "",
) -> dict[pd.Timestamp, tuple[float, np.ndarray]]:
    """
    Estimation roulante (fenêtre 60 mois) des paramètres (ν_c, γ_c) de la copule.
    Warm-starting entre dates consécutives.
    """
    copula_params: dict = {}
    nu0     = 8.0
    gamma0  = np.zeros(N_FACTORS)
    all_dt  = pit_df.index

    for date_t in tqdm(oos_dates, desc=f"  Copule {label}", leave=False):
        idx_end   = np.searchsorted(all_dt, date_t, side="right") - 1
        idx_start = max(0, idx_end - WINDOW_COPULA + 1)
        win_dates = all_dt[idx_start : idx_end + 1]

        if len(win_dates) < 12:
            copula_params[date_t] = (nu0, gamma0.copy())
            continue

        u_win = pit_df.reindex(win_dates)[FACTEURS].values
        u_win = np.clip(np.where(np.isfinite(u_win), u_win, 0.5), 1e-6, 1.0 - 1e-6)
        R_win = _build_R_window(corr_df_eom, win_dates)

        try:
            nu_c, gamma_c = estimate_copula_params(u_win, R_win, nu0=nu0, gamma0=gamma0)
        except Exception:
            nu_c, gamma_c = nu0, gamma0.copy()

        copula_params[date_t] = (nu_c, gamma_c)
        nu0    = nu_c
        gamma0 = gamma_c.copy()

    return copula_params


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 : OPTIMISATION CVaR — MAXIMISATION DIRECTE DU RATIO RETURN/CVaR
# ──────────────────────────────────────────────────────────────────────────────

def _tail_mean(losses: np.ndarray, beta: float) -> float:
    """
    CVaR empirique via sélection partielle O(Q), plus rapide qu'un quantile complet.
    On moyenne les plus grandes pertes dans la queue 1-β.
    """
    q = losses.shape[0]
    tail_count = max(1, int(np.ceil((1.0 - beta) * q)))
    kth = q - tail_count
    tail = np.partition(losses, kth)[kth:]
    return float(tail.mean())


def _portfolio_cvar(w: np.ndarray, r_sim: np.ndarray, beta: float) -> float:
    """
    Calcule la CVaR_β empirique sur Q rendements simulés.
    CVaR = E[-r_p | -r_p ≥ VaR_β]  (convention perte positive).
    """
    losses = -(r_sim @ w)
    return _tail_mean(losses, beta)


def _append_unique_start(starts: list[np.ndarray], candidate: np.ndarray | None) -> None:
    """Ajoute un point de départ valide si aucun start équivalent n'existe déjà."""
    if candidate is None:
        return
    cand = np.asarray(candidate, dtype=float)
    if not np.all(np.isfinite(cand)):
        return
    for start in starts:
        if np.allclose(start, cand, atol=1e-10, rtol=0.0):
            return
    starts.append(cand.copy())


def _weights_to_13030_start(w: np.ndarray) -> np.ndarray:
    """Reconstruit un état (longs, shorts) admissible à partir des poids nets."""
    longs = np.maximum(w, 0.0)
    shorts = np.maximum(-w, 0.0)

    if longs.sum() <= 1e-12:
        longs = np.full_like(longs, LONG_TARGET / len(w))
    else:
        longs *= LONG_TARGET / longs.sum()

    if shorts.sum() <= 1e-12:
        shorts = np.full_like(shorts, SHORT_TARGET / len(w))
    else:
        shorts *= SHORT_TARGET / shorts.sum()

    return np.concatenate([longs, shorts])


def _max_ratio_longonly(
    r_sim: np.ndarray,
    mu_t: np.ndarray,
    beta: float,
    rf_t: float,
    warm_start: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Portefeuille tangent long-only : argmax (μ_p − rf) / CVaR_β(p).

    Section 5.2 du papier : "the one with higher Sharpe ratio or return/CVaR
    ratio in the frontier". Ici maximisation directe via SLSQP.
    CVaR est quasi-convexe en w, donc return/CVaR est quasi-concave : SLSQP
    converge vers l'optimum global avec plusieurs initialisations.

    Retourne
    --------
    (w_opt, ratio_opt, cvar_opt)
    """
    N = len(mu_t)

    def neg_ratio(w: np.ndarray) -> float:
        cvar = _portfolio_cvar(w, r_sim, beta)
        if cvar < 1e-10:
            return 1e6
        return -(float(mu_t @ w) - rf_t) / cvar

    constraints = [{"type": "eq", "fun": lambda w: float(w.sum()) - 1.0}]
    bounds      = [(0.0, 1.0)] * N

    starts: list[np.ndarray] = []
    _append_unique_start(starts, warm_start)
    _append_unique_start(starts, np.ones(N) / N)
    for i in range(N):
        _append_unique_start(starts, np.eye(N)[i] * 0.6 + np.ones(N) * 0.4 / N)

    best_ratio = -np.inf
    best_w     = np.ones(N) / N

    for w0 in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    neg_ratio, w0, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER_LO},
                )
            if res.success or res.status == 9:   # 9 = iteration limit (acceptable)
                ratio = -float(res.fun)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_w     = np.clip(res.x, 0.0, 1.0)
                    best_w    /= best_w.sum()      # re-normalise
        except Exception:
            continue

    cvar_opt = _portfolio_cvar(best_w, r_sim, beta)
    return best_w, best_ratio, cvar_opt


def _max_ratio_130_30(
    r_sim: np.ndarray,
    mu_t: np.ndarray,
    beta: float,
    rf_t: float,
    warm_start: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Portefeuille tangent 130/30 : argmax (μ_p − rf) / CVaR_β(p).
    Décomposition w = l − s, sum(l)=1.30, sum(s)=0.30, l≥0, s≥0.
    """
    N = len(mu_t)

    def neg_ratio_130(x: np.ndarray) -> float:
        w    = x[:N] - x[N:]
        cvar = _portfolio_cvar(w, r_sim, beta)
        if cvar < 1e-10:
            return 1e6
        return -(float(mu_t @ w) - rf_t) / cvar

    constraints = [
        {"type": "eq", "fun": lambda x: float(x[:N].sum()) - LONG_TARGET},
        {"type": "eq", "fun": lambda x: float(x[N:].sum()) - SHORT_TARGET},
    ]
    bounds = [(0.0, None)] * (2 * N)

    x0_base = np.concatenate([
        np.full(N, LONG_TARGET / N),
        np.full(N, SHORT_TARGET / N),
    ])

    # 3 points de départ (base + 2 perturbations)
    rng_starts = np.random.default_rng(0)
    starts: list[np.ndarray] = []
    _append_unique_start(starts, warm_start)
    _append_unique_start(starts, x0_base)
    for _ in range(2):
        noise = rng_starts.normal(0, 0.02, 2 * N)
        x_pert = x0_base + noise
        x_pert[:N]  = np.maximum(x_pert[:N], 0)
        x_pert[:N] *= LONG_TARGET / x_pert[:N].sum()
        x_pert[N:]  = np.maximum(x_pert[N:], 0)
        x_pert[N:] *= SHORT_TARGET / x_pert[N:].sum()
        _append_unique_start(starts, x_pert)

    best_ratio = -np.inf
    best_w     = np.full(N, LONG_TARGET / N) - np.full(N, SHORT_TARGET / N)

    for x0 in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    neg_ratio_130, x0, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER_13030},
                )
            if res.success or res.status == 9:
                ratio = -float(res.fun)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_w     = res.x[:N] - res.x[N:]
        except Exception:
            continue

    cvar_opt = _portfolio_cvar(best_w, r_sim, beta)
    return best_w, best_ratio, cvar_opt


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 : MÉTRIQUES DE PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    returns: pd.Series,
    rf: pd.Series,
    cvar_series: pd.Series,
) -> dict:
    """
    Métriques des Tables 9 et 10 :
        - Rendement annualisé (%)
        - Return/CVaR (annualisé)
        - Ratio de Sortino (annualisé)
        - Maximum Drawdown (%)
    """
    r = returns.dropna()
    if len(r) < 2:
        return {}

    rf_al  = rf.reindex(r.index).fillna(0.0)
    excess = r - rf_al

    # Rendement annualisé : log-rendement moyen × 12 → rendement simple
    ann_ret = float(np.expm1(r.mean() * 12.0)) * 100.0

    # Return/CVaR
    cvar_al = cvar_series.reindex(r.index).dropna()
    if len(cvar_al) > 0 and float(cvar_al.mean()) > 1e-10:
        ret_cvar = float(r.reindex(cvar_al.index).mean()) / float(cvar_al.mean())
    else:
        ret_cvar = np.nan

    # Sortino ratio annualisé
    neg_ex = excess[excess < 0]
    std_neg = float(neg_ex.std(ddof=1)) if len(neg_ex) > 1 else np.nan
    sortino = float(excess.mean() / std_neg) * np.sqrt(12.0) if (std_neg and std_neg > 0) else np.nan

    # Maximum Drawdown
    cum = np.exp(r.cumsum())
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min()) * 100.0

    return {
        "ann_return_pct": ann_ret,
        "return_cvar":    ret_cvar,
        "sortino":        sortino,
        "mdd_pct":        mdd,
        "n_obs":          len(r),
    }


def _compute_sharpe_ratio(returns: pd.Series, rf: pd.Series) -> float:
    """Sharpe ratio OOS mensuel, utilisé pour identifier le meilleur tangent."""
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    excess = r - rf.reindex(r.index).fillna(0.0)
    std = float(excess.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return np.nan
    return float(excess.mean()) / std


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 : PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def run_meancvar_optimization(verbose: bool = True) -> None:
    """Pipeline complet — Section 5.2 du papier."""
    if verbose:
        print("=" * 70)
        print("SECTION 5.2 — OPTIMISATION MOYENNE-CVaR (copule GH skewed-t)")
        print("Zhao et al. (2019) — Tables 9 et 10")
        print(f"Q={N_SIMS} simulations, β={BETAS}, méthode=SLSQP direct")
        print("=" * 70)

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    DIR_RESULTS_CVAR.mkdir(parents=True, exist_ok=True)
    DIR_RESULTS_95.mkdir(parents=True, exist_ok=True)
    DIR_RESULTS_99.mkdir(parents=True, exist_ok=True)
    DIR_RESULTS_BEST.mkdir(parents=True, exist_ok=True)

    # ── 1. Données ───────────────────────────────────────────────────────────
    if verbose:
        print("\n[1/6] Chargement des données...")

    monthly   = _charger_rendements()
    rf_series = _charger_rf()
    pit_df    = _charger_pit()
    resid_df  = _charger_residus()

    mask_oos    = (monthly.index >= OOS_START) & (monthly.index <= OOS_END)
    monthly_oos = monthly.loc[mask_oos].copy()
    rf_oos      = rf_series.reindex(monthly_oos.index).fillna(0.0)
    oos_dates   = monthly_oos.index

    if verbose:
        print(f"  OOS : {oos_dates[0].date()} → {oos_dates[-1].date()} ({len(oos_dates)} mois)")

    rw_prev    = _charger_previsions_rw(monthly, oos_dates)
    svr_prev   = _charger_previsions(CHEMIN_SVR).reindex(oos_dates)
    scsvr_prev = _charger_previsions(CHEMIN_SC_SVR).reindex(oos_dates)
    dma_prev   = _charger_previsions(CHEMIN_DMA).reindex(oos_dates)

    dcc_eom  = _align_corr_to_eom(pd.read_parquet(CHEMIN_DCC))
    adcc_eom = _align_corr_to_eom(pd.read_parquet(CHEMIN_ADCC))
    gas_eom  = _align_corr_to_eom(pd.read_parquet(CHEMIN_GAS))

    modeles_prev = {"RW": rw_prev, "SVR": svr_prev, "SC-SVR": scsvr_prev, "DMA": dma_prev}
    modeles_corr = {"DCC": dcc_eom, "ADCC": adcc_eom, "GAS": gas_eom}
    prev_arrays = {
        name: df.reindex(oos_dates)[FACTEURS].to_numpy(dtype=float)
        for name, df in modeles_prev.items()
    }
    corr_aligned = {
        name: df.reindex(oos_dates)
        for name, df in modeles_corr.items()
    }
    rf_arr = rf_oos.reindex(oos_dates).to_numpy(dtype=float)
    real_arr = monthly_oos[FACTEURS].to_numpy(dtype=float)

    # ── 2. Paramètres marginaux Hansen ───────────────────────────────────────
    if verbose:
        print("\n[2/6] Paramètres marginaux Hansen skewed-t...")
    marginal_params = _estimate_marginal_params(resid_df)
    if verbose:
        for f, (nu, lam) in zip(FACTEURS, marginal_params):
            print(f"  {f}: ν={nu:.2f}  λ={lam:+.4f}")

    # ── 3. Volatilités GARCH rolling ─────────────────────────────────────────
    if verbose:
        print("\n[3/6] Volatilités GARCH(1,1) rolling (cache en résultats)...")
    sigma_t = _compute_garch_vol_rolling(monthly, oos_dates, verbose=verbose)
    sigma_arr = sigma_t.reindex(oos_dates)[FACTEURS].to_numpy(dtype=float)
    if verbose:
        ok = sigma_t.notna().all(axis=1).sum()
        print(f"  Complètes : {ok}/{len(oos_dates)} mois")

    # ── 4. Paramètres de copule roulants ─────────────────────────────────────
    if verbose:
        print("\n[4/6] Estimation roulante copule GH skewed-t (60 mois)...")
    copula_by_corr: dict[str, dict] = {}
    for corr_name, corr_df in modeles_corr.items():
        if verbose:
            print(f"  {corr_name}...")
        copula_by_corr[corr_name] = _estimate_copula_rolling(
            pit_df, corr_df, oos_dates, label=corr_name
        )

    # ── 5. Boucle principale ─────────────────────────────────────────────────
    if verbose:
        print("\n[5/6] Optimisation moyenne-CVaR...")
        print("  Mode rapide actif : SLSQP direct, aucun LP, 1 simulation par (corr_model, date)")

    # Initialisation des conteneurs de résultats
    # Clé : "{prev}-{corr}-SKT-{beta_pct}-{panel}"
    weights_store: dict[str, dict] = {}
    returns_store: dict[str, dict] = {}
    cvar_store:    dict[str, dict] = {}

    for prev_n in modeles_prev:
        for corr_n in modeles_corr:
            for beta in BETAS:
                for panel in ["lo", "130"]:
                    k = f"{prev_n}-{corr_n}-SKT-{int(beta*100)}-{panel}"
                    weights_store[k] = {}
                    returns_store[k] = {}
                    cvar_store[k]    = {}

    seed_gen = np.random.default_rng(GLOBAL_SEED)
    warm_lo: dict[tuple[str, str, float], np.ndarray] = {}
    warm_130: dict[tuple[str, str, float], np.ndarray] = {}

    # Boucle externe : modèle de corrélation (3 × 211 simulations)
    for corr_name, corr_df in corr_aligned.items():
        cop_dict = copula_by_corr[corr_name]

        for idx_t, date_t in enumerate(tqdm(oos_dates, desc=f"  {corr_name}", leave=True)):

            # ── Données communes à la date t ──────────────────────────────
            sig_vec = sigma_arr[idx_t]
            rf_t = float(rf_arr[idx_t]) if np.isfinite(rf_arr[idx_t]) else 0.0
            r_real = real_arr[idx_t]

            if np.any(~np.isfinite(sig_vec)) or np.any(sig_vec <= 0) or np.any(~np.isfinite(r_real)):
                continue

            corr_row = corr_df.iloc[idx_t]
            if corr_row.isna().all():
                continue
            R_t = corr_row_to_matrix(corr_row, FACTEURS_CORR)

            nu_c, gamma_c = cop_dict.get(date_t, (8.0, np.zeros(N_FACTORS)))
            seed_t = int(seed_gen.integers(0, 2**31))

            # ── SIMULATION UNIQUE avec mu=0 (résidus purs) ────────────────
            # r_zero = sigma_t * z  (sans la moyenne de prévision)
            # Partagé entre tous les modèles de prévision : 4× moins de simulations
            try:
                r_zero = simulate_portfolio_returns(
                    mu_t=np.zeros(N_FACTORS),
                    sigma_t=sig_vec,
                    R_t=R_t,
                    nu_c=nu_c,
                    gamma_c=gamma_c,
                    marginal_params=marginal_params,
                    n_sims=N_SIMS,
                    seed=seed_t,
                )
            except Exception:
                continue

            # ── Boucle sur les modèles de prévision ───────────────────────
            for prev_name, prev_arr in prev_arrays.items():
                mu_vec = prev_arr[idx_t]
                if np.any(~np.isfinite(mu_vec)):
                    continue

                # Rendements simulés = résidus + prévision
                r_sim = r_zero + mu_vec[np.newaxis, :]   # (Q, N)

                # ── Optimisation pour chaque (β, panel) ───────────────────
                for beta in BETAS:
                    # Panel B : long-only
                    k_lo = f"{prev_name}-{corr_name}-SKT-{int(beta*100)}-lo"
                    warm_key = (prev_name, corr_name, beta)
                    w_lo, ratio_lo, cvar_lo = _max_ratio_longonly(
                        r_sim, mu_vec, beta, rf_t, warm_start=warm_lo.get(warm_key)
                    )
                    if np.isfinite(ratio_lo):
                        warm_lo[warm_key] = w_lo.copy()
                        weights_store[k_lo][date_t] = w_lo
                        returns_store[k_lo][date_t] = float(r_real @ w_lo)
                        cvar_store[k_lo][date_t]    = cvar_lo

                    # Panel C : 130/30
                    k_130 = f"{prev_name}-{corr_name}-SKT-{int(beta*100)}-130"
                    w_130, ratio_130, cvar_130 = _max_ratio_130_30(
                        r_sim, mu_vec, beta, rf_t, warm_start=warm_130.get(warm_key)
                    )
                    if np.isfinite(ratio_130):
                        warm_130[warm_key] = _weights_to_13030_start(w_130)
                        weights_store[k_130][date_t] = w_130
                        returns_store[k_130][date_t] = float(r_real @ w_130)
                        cvar_store[k_130][date_t]    = cvar_130

    # ── 6. Sauvegarde ─────────────────────────────────────────────────────────
    if verbose:
        print("\n[6/6] Sauvegarde des résultats...")

    _save_results(
        weights_store, returns_store, cvar_store,
        monthly_oos, rf_oos, modeles_prev, modeles_corr,
        verbose=verbose,
    )

    # ── Notes de réplication ──────────────────────────────────────────────────
    _write_replication_notes(marginal_params=marginal_params, n_oos=len(oos_dates))

    if verbose:
        print("\n" + "=" * 70)
        print("SECTION 5.2 TERMINÉE")
        print(f"Résultats : {DIR_RESULTS_CVAR}")
        print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 : SAUVEGARDE DES RÉSULTATS
# ──────────────────────────────────────────────────────────────────────────────

def _save_results(
    weights_store: dict,
    returns_store: dict,
    cvar_store: dict,
    monthly_oos: pd.DataFrame,
    rf_oos: pd.Series,
    modeles_prev: dict,
    modeles_corr: dict,
    verbose: bool = True,
) -> None:
    """Sauvegarde les 8 CSV de poids/rendements + résumé de performance."""

    N      = N_FACTORS
    oos_dates = monthly_oos.index
    DIR_RESULTS_CVAR.mkdir(parents=True, exist_ok=True)
    DIR_RESULTS_95.mkdir(parents=True, exist_ok=True)
    DIR_RESULTS_99.mkdir(parents=True, exist_ok=True)
    DIR_RESULTS_BEST.mkdir(parents=True, exist_ok=True)

    # Benchmark 1/N
    w_1n = np.ones(N) / N
    r_1n = monthly_oos[FACTEURS].values @ w_1n
    pd.DataFrame(
        np.tile(w_1n, (len(oos_dates), 1)), index=oos_dates, columns=FACTEURS
    ).to_csv(DIR_RESULTS_CVAR / "mean_cvar_weights_1N.csv", date_format="%Y-%m-%d")
    pd.Series(r_1n, index=oos_dates, name="portfolio_return").to_csv(
        DIR_RESULTS_CVAR / "mean_cvar_returns_1N.csv", date_format="%Y-%m-%d"
    )

    summary_rows = []

    combos = [(p, c) for p in modeles_prev for c in modeles_corr]

    for beta in BETAS:
        beta_str = int(beta * 100)
        beta_dir = DIR_RESULTS_95 if beta_str == 95 else DIR_RESULTS_99
        for panel_sfx, panel_label in [("lo", "long_only"), ("130", "130_30")]:
            all_w, all_r = [], []
            best_sharpe = -np.inf
            best_label = None
            best_w_df = None
            best_r_df = None

            for (prev_n, corr_n) in combos:
                key   = f"{prev_n}-{corr_n}-SKT-{beta_str}-{panel_sfx}"
                label = f"{prev_n}-{corr_n}-SKT"

                w_dict = weights_store.get(key, {})
                r_dict = returns_store.get(key, {})
                c_dict = cvar_store.get(key, {})
                if not w_dict:
                    continue

                w_df = pd.DataFrame.from_dict(w_dict, orient="index", columns=FACTEURS)
                r_s  = pd.Series(r_dict, name="portfolio_return")
                c_s  = pd.Series(c_dict, name="portfolio_cvar")

                # CSV poids
                row_w = w_df.copy()
                row_w.insert(0, "model", label)
                all_w.append(row_w)

                # CSV rendements
                row_r = pd.DataFrame({"model": label, "portfolio_return": r_s, "portfolio_cvar": c_s})
                all_r.append(row_r)

                sharpe = _compute_sharpe_ratio(r_s, rf_oos)
                if np.isfinite(sharpe) and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_label = label
                    best_w_df = row_w.copy()
                    best_r_df = row_r.copy()

                # Métriques
                m = _compute_metrics(r_s, rf_oos, c_s)
                if m:
                    summary_rows.append({"model": label, "beta": beta, "panel": panel_label, **m})
                    if verbose:
                        print(f"  {label:35s} β={beta:.2f} {panel_label:10s} "
                              f"ret={m.get('ann_return_pct', np.nan):+6.2f}%  "
                              f"R/CVaR={m.get('return_cvar', np.nan):.3f}  "
                              f"MDD={m.get('mdd_pct', np.nan):+6.2f}%")

            fname_w = f"mean_cvar_{beta_str}_weights_{panel_label}.csv"
            fname_r = f"mean_cvar_{beta_str}_returns_{panel_label}.csv"
            if all_w:
                pd.concat(all_w).to_csv(beta_dir / fname_w, date_format="%Y-%m-%d")
            if all_r:
                pd.concat(all_r).to_csv(beta_dir / fname_r, date_format="%Y-%m-%d")
            if best_w_df is not None and best_r_df is not None:
                best_w_name = f"best_tangent_{beta_str}_weights_{panel_label}.csv"
                best_r_name = f"best_tangent_{beta_str}_returns_{panel_label}.csv"
                best_w_df.to_csv(DIR_RESULTS_BEST / best_w_name, date_format="%Y-%m-%d")
                best_r_df.to_csv(DIR_RESULTS_BEST / best_r_name, date_format="%Y-%m-%d")
            if verbose:
                print(f"  → {beta_dir.name}/{fname_w}, {beta_dir.name}/{fname_r}")
                if best_label is not None:
                    print(f"  → best_portfolio_tangent/{best_w_name}, {best_r_name} [{best_label}]")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            DIR_RESULTS_CVAR / "mean_cvar_performance_summary.csv", index=False
        )
        if verbose:
            print("  → mean_cvar_performance_summary.csv")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 : NOTES DE RÉPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def _write_replication_notes(
    marginal_params: list[tuple[float, float]],
    n_oos: int,
) -> None:
    notes = f"""================================================================================
NOTES DE RÉPLICATION — Section 5.2 (Mean-CVaR copule GH skewed-t)
Zhao, Stasinakis, Sermpinis & Da Silva Fernandes (2019)
Int J Fin Econ, 24, 1443-1463
================================================================================
DATE : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

── FICHIERS ────────────────────────────────────────────────────────────────────
  Rendements  : data/monthly_log_returns.csv
  RF          : data/F-F_Research_Data_5_Factors_2x3.csv
  Prévisions  : previsions_svr.csv, previsions_sc_svr.csv, previsions_dma.csv
  Corrélations: correlations_dcc/adcc/gas.parquet
  PIT         : results/ext/uniforms_pit.parquet
  Résidus     : results/ext/residuals_garch.parquet
  Cache GARCH : results/optimization/garch_vols_cache.csv

── DIMENSIONS ──────────────────────────────────────────────────────────────────
  Facteurs : MKT, SMB, HML, RMW, CMA (N=5)
  OOS      : 2000-01-31 → 2017-07-31 ({n_oos} mois)
  Q        : {N_SIMS} simulations Monte Carlo
  Fenêtre  : {WINDOW_COPULA} mois (Section 5.2, note 10)
  Niveaux β: {BETAS}

── PARAMÈTRES MARGINAUX HANSEN (1994) ──────────────────────────────────────────
""" + "".join(
        f"  {f:8s}: ν={nu:.4f}  λ={lam:+.4f}\n"
        for f, (nu, lam) in zip(FACTEURS, marginal_params)
    ) + f"""
── COPULE GH SKEWED-T (Demarta & McNeil 2005, Appendice D) ─────────────────────
  Construction : X = γ_c W + √W Z,  W~InvGamma(ν_c/2, ν_c/2),  Z~N(0,R_t)
  CDF marginale: quadrature Gauss-Laguerre généralisée (40 nœuds)
  Estimation   : IFM 2 étapes (MLE ν_c via copule t, puis γ_c par mom.)
  Corrélation  : R_t de DCC/ADCC/GAS fournie directement

── OPTIMISATION ────────────────────────────────────────────────────────────────
  Objectif   : max (μ_p − rf) / CVaR_β  (Section 5.2 : portefeuille tangent)
  Méthode    : SLSQP multi-départ (6 starts long-only, 3 starts 130/30)
  Alternative aux papier : sweep frontière N points → 1 optimisation SLSQP
  Justification : return/CVaR est quasi-concave → SLSQP trouve le global optimum
  Stratégie loop: simulation UNIQUE par (corr_model, date), partagée entre
                  les 4 prévisions → 4× moins de simulations

── PRÉDICTEUR "BEST" ────────────────────────────────────────────────────────────
  Non calculé : best_individual.csv couvre uniquement IS (1965-1999).

── ÉCARTS POTENTIELS AVEC LE PAPIER ────────────────────────────────────────────
  1. Estimation copule : IFM 2 étapes vs MLE conjointe complète
  2. SLSQP vs sweep de frontière LP (résultats équivalents, méthode différente)
  3. Prédicteur "Best" absent (données OOS non disponibles)

── COMPARAISON QUALITATIVE ATTENDUE (Tables 9 et 10) ───────────────────────────
  DMA > SC-SVR > SVR > RW   (return/CVaR, Sortino)
  GAS > ADCC > DCC           (modèle de corrélation)
  130/30 > long-only         (grâce à la vente à découvert)
  DMA-GAS-SKT best : return/CVaR ≈ 2.9-3.0, MDD ≈ 9.2%
================================================================================
"""
    with open(DIR_RESULTS / "mean_cvar_replication_notes.txt", "w", encoding="utf-8") as f:
        f.write(notes)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_meancvar_optimization(verbose=True)
