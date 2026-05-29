#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
# src/04_portfolio/mv_optimization.py

"""
Section 5.1 — Optimisation Moyenne-Variance (M-V)
Réplication fidèle de la Section 5.1 du papier :
    Zhao, Stasinakis, Sermpinis & Da Silva Fernandes (2019)
    "Revisiting Fama-French factors' predictability with Bayesian modelling
     and copula-based portfolio optimization"
    International Journal of Finance & Economics, 24, 1443-1463.

Pipeline complet :
    1. Chargement des données (rendements, prévisions, matrices de corrélation)
    2. Calcul des volatilités conditionnelles GARCH(1,1) rolling (fenêtre 60 mois)
    3. Reconstruction de la matrice de covariance : Sigma_t = D_t * R_t * D_t
    4. Frontière efficiente + portefeuille tangent (max Sharpe) — long-only
    5. Frontière efficiente + portefeuille tangent (max Sharpe) — 130/30
    6. Benchmark 1/N (portefeuille équipondéré)
    7. Sauvegarde des résultats

Combinaisons testées (Section 5.1, Table 8 du papier) :
    Prévisions : RW | SC-SVR | DMA
    Covariance : DCC-GARCH | ADCC-GARCH | GAS

Conventions :
    - Les log-rendements sont dans monthly_log_returns.csv (unité décimale).
    - Les matrices de corrélation sont indexées en début de mois (2000-01-01),
      converties en fin de mois pour alignement avec les rendements.
    - Le taux sans risque provient du fichier Ken French (RF en %).
    - Les volatilités GARCH(1,1) sont estimées sur des fenêtres roulantes de 60
      mois, en accord avec garch.py (src/dependance/garch.py).
    - La stratégie 130/30 est implémentée par décomposition : w = l - s,
      avec sum(l) = 1.30 et sum(s) = 0.30, l >= 0, s >= 0.

Sorties :
    results/optimization/           — toutes les séries de poids et rendements
    results/optimization/best_portfolio_tangent/ — portefeuilles tangents (CSV)
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

# ==============================================================================
# CHEMINS ET CONSTANTES
# ==============================================================================

ROOT = Path(__file__).resolve().parents[2]

CHEMIN_RETURNS = ROOT / "data" / "monthly_log_returns.csv"
CHEMIN_FF_RAW  = ROOT / "data" / "F-F_Research_Data_5_Factors_2x3.csv"
CHEMIN_DMA     = ROOT / "data" / "02_forecasting" / "previsions_dma.csv"
CHEMIN_SC_SVR  = ROOT / "data" / "02_forecasting" / "previsions_sc_svr.csv"
CHEMIN_DCC     = ROOT / "results" / "ext" / "correlations_dcc.parquet"
CHEMIN_ADCC    = ROOT / "results" / "ext" / "correlations_adcc.parquet"
CHEMIN_GAS     = ROOT / "results" / "ext" / "correlations_gas.parquet"

DIR_RESULTS = ROOT / "results" / "optimization"
DIR_BEST    = ROOT / "results" / "optimization" / "best_portfolio_tangent"

# Noms des 5 facteurs Fama-French (papier Section 2)
FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]

# Les matrices de corrélation DCC/ADCC/GAS utilisent "MKT_RF" au lieu de "MKT"
FACTEURS_CORR = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]

# Fenêtre roulante : 60 mois (5 ans), cohérent avec garch.py et Section 5.2 du papier
WINDOW_GARCH = 60

# Nombre de points sur la frontière efficiente
N_FRONTIER = 200

# Levier 130/30 (papier Section 6, note de bas de page 11)
LONG_TARGET  = 1.30   # exposition longue totale
SHORT_TARGET = 0.30   # exposition courte totale (valeur absolue)


# ==============================================================================
# SECTION 1 : CHARGEMENT DES DONNEES
# ==============================================================================

def _charger_rendements() -> pd.DataFrame:
    """
    Charge les log-rendements mensuels des 5 facteurs (1965-01 à 2017-07).
    Source : data/monthly_log_returns.csv
    Unité  : log-rendements décimaux, ex. 0.034 = 3.4%/mois en log-rendement.
    """
    df = pd.read_csv(
        CHEMIN_RETURNS, index_col=0, parse_dates=True, date_format="%Y-%m-%d"
    )
    return df[FACTEURS]


def _charger_rf() -> pd.Series:
    """
    Charge le taux sans risque mensuel de Ken French et le convertit en
    log-rendement décimal.
    Le fichier FF donne RF en % (ex. 0.41 = 0.41%/mois).
    """
    ff = pd.read_csv(CHEMIN_FF_RAW, skiprows=4, index_col=0, dtype=str)
    ff = ff[ff.index.str.match(r"^\d{6}$", na=False)].copy()
    rf = ff[["RF"]].apply(pd.to_numeric, errors="coerce").dropna()
    rf.index = pd.to_datetime(rf.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    rf.index.name = "date"
    # Conversion : % -> log-rendement
    rf_log = np.log(1.0 + rf["RF"] / 100.0)
    return rf_log


def _charger_previsions_rw(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Prévisions Random Walk (RW) : prévision pour t = rendement réalisé en t-1.
    Cohérent avec evaluation.py : charger_previsions_rw().
    """
    df = monthly[monthly.index <= OOS_END].shift(1)
    masque = (df.index >= OOS_START) & (df.index <= OOS_END)
    return df.loc[masque, FACTEURS]


def _charger_previsions(chemin: Path) -> pd.DataFrame:
    """Charge un fichier de prévisions OOS (SC-SVR ou DMA)."""
    df = pd.read_csv(chemin, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    return df[FACTEURS]


def _charger_correlations(chemin: Path) -> pd.DataFrame:
    """Charge une matrice de corrélation depuis un fichier parquet."""
    return pd.read_parquet(chemin)


def _align_corr_to_eom(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Les matrices de corrélation DCC/ADCC/GAS sont indexées en début de mois
    (ex. 2000-01-01). Les rendements et prévisions sont en fin de mois
    (ex. 2000-01-31). Cette fonction convertit l'index pour aligner par
    année-mois.
    Convention : la corrélation du 2000-01-01 correspond au mois de janvier 2000,
    donc à la réalisation du 2000-01-31.
    """
    df = corr.copy()
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
    return df


# ==============================================================================
# SECTION 2 : VOLATILITES CONDITIONNELLES GARCH(1,1) ROLLING
# ==============================================================================

def _compute_garch_vol_rolling(
    monthly: pd.DataFrame,
    oos_dates: pd.DatetimeIndex,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calcule la volatilité conditionnelle GARCH(1,1) rolling pour chaque facteur.

    Méthode (cohérente avec garch.py, Appendice B du papier) :
        - Pour chaque date t dans oos_dates, on estime GARCH(1,1) sur la
          fenêtre [t - 60 mois + 1 : t] incluse (60 mois).
        - On extrait conditional_volatility.iloc[-1] comme sigma_{i,t}.
        - Le modèle : GARCH(1,1) avec moyenne nulle (modèle AR est traité
          séparément dans garch.py, ici on vise uniquement sigma_t).

    Unité de sigma_{i,t} : même unité que monthly (log-rendement décimal).

    En cas d'échec du GARCH, fallback sur l'écart-type empirique de la fenêtre.

    Paramètres
    ----------
    monthly   : pd.DataFrame — log-rendements mensuels complets (pré-OOS inclus)
    oos_dates : pd.DatetimeIndex — dates OOS (2000-01-31 à 2017-07-31)
    verbose   : bool

    Retourne
    --------
    pd.DataFrame (T_oos × 5) — sigma_{i,t} pour chaque facteur et date OOS
    """
    vols = pd.DataFrame(index=oos_dates, columns=FACTEURS, dtype=float)

    for facteur in FACTEURS:
        serie = monthly[facteur]
        n_success = 0
        n_fallback = 0

        for date_t in tqdm(oos_dates, desc=f"  GARCH vol {facteur}", leave=False):
            # Localiser date_t dans la série complète
            try:
                idx_t = serie.index.get_loc(date_t)
            except KeyError:
                vols.at[date_t, facteur] = np.nan
                continue

            # Fenêtre de 60 mois incluse : [t-59 : t] (indexation Python)
            start = max(0, idx_t - WINDOW_GARCH + 1)
            window = serie.iloc[start : idx_t + 1]

            if len(window) < 10:
                vols.at[date_t, facteur] = float(window.std()) if len(window) > 1 else np.nan
                n_fallback += 1
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Entrée en log-rendements décimaux (ex. 0.034)
                    # arch_model est plus stable si on rescale en % ou bps
                    # On passe en pourcentage pour la stabilité numérique de arch,
                    # puis on divise le résultat par 100 pour revenir en décimal
                    model = arch_model(
                        window * 100.0,
                        mean="Zero",
                        vol="GARCH",
                        p=1,
                        q=1,
                        dist="normal",
                        rescale=False,
                    )
                    result = model.fit(disp="off", show_warning=False)
                    # conditional_volatility est en % (même unité que l'input)
                    sigma_pct = float(result.conditional_volatility.iloc[-1])
                    sigma_t   = sigma_pct / 100.0  # retour en décimal
                    vols.at[date_t, facteur] = sigma_t
                    n_success += 1

            except Exception:
                # Fallback : écart-type empirique de la fenêtre (en décimal)
                vols.at[date_t, facteur] = float(window.std())
                n_fallback += 1

        if verbose:
            print(
                f"    {facteur} : {n_success} GARCH réussis, "
                f"{n_fallback} fallback std empirique"
            )

    return vols.astype(float)


# ==============================================================================
# SECTION 3 : MATRICE DE COVARIANCE Sigma_t = D_t * R_t * D_t
# ==============================================================================

def _corr_row_to_matrix(row: pd.Series) -> np.ndarray:
    """
    Reconstruit la matrice de corrélation (5×5) à partir d'une ligne de paires
    au format 'FACTEUR_A-FACTEUR_B' (triangle supérieur du parquet).

    Les corrélations DCC/ADCC/GAS utilisent 'MKT_RF' pour le facteur de marché.
    On mappe MKT_RF → MKT (premier élément de FACTEURS).

    En cas de valeur manquante ou de matrice non-PSD, on projette sur le cône
    des matrices semi-définies positives (correction spectrale minimale).
    """
    N = len(FACTEURS)
    R = np.eye(N)

    for i, fi in enumerate(FACTEURS_CORR):
        for j, fj in enumerate(FACTEURS_CORR):
            if j <= i:
                continue
            pair1 = f"{fi}-{fj}"
            pair2 = f"{fj}-{fi}"
            if pair1 in row.index:
                rho = row[pair1]
            elif pair2 in row.index:
                rho = row[pair2]
            else:
                rho = np.nan

            if pd.isna(rho):
                rho = 0.0
            rho = float(np.clip(rho, -1.0 + 1e-6, 1.0 - 1e-6))
            R[i, j] = rho
            R[j, i] = rho

    # Symétrie exacte
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 1.0)

    # Projection PSD : correction spectrale minimale si nécessaire
    eigvals = np.linalg.eigvalsh(R)
    if np.any(eigvals < 1e-8):
        R += (1e-8 - eigvals.min()) * np.eye(N)
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        np.fill_diagonal(R, 1.0)

    return R


def _build_sigma(R: np.ndarray, sigma_vec: np.ndarray) -> np.ndarray:
    """
    Construit la matrice de covariance conditionnelle :
        Sigma_t = D_t * R_t * D_t
    où D_t = diag(sigma_{1,t}, ..., sigma_{5,t}).

    La correction PSD garantit que Sigma_t est inversible.
    """
    D = np.diag(sigma_vec)
    Sigma = D @ R @ D

    # Symétrie exacte
    Sigma = (Sigma + Sigma.T) / 2.0

    # Correction PSD
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.any(eigvals < 1e-12):
        Sigma += (1e-12 - eigvals.min()) * np.eye(len(sigma_vec))

    return Sigma


# ==============================================================================
# SECTION 4 : OPTIMISATION MOYENNE-VARIANCE
# ==============================================================================

def _mv_longonly(
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float,
    n_points: int = None,  # conservé pour compatibilité, non utilisé
) -> tuple[np.ndarray, float]:
    """
    Portefeuille tangent long-only par maximisation directe du Sharpe ratio.

    Problème (équation (11) du papier) :
        max_w  (w' mu - rf) / sqrt(w' Sigma w)
        s.t.   w' 1 = 1,  w_i >= 0   (long-only, Panel B)

    Stratégie multi-départ : 5 points d'initialisation (équipondéré + 4
    concentrations unitaires) pour éviter les optima locaux avec N=5.

    Paramètres
    ----------
    mu    : np.ndarray (N,) — rendements attendus des facteurs
    Sigma : np.ndarray (N,N) — matrice de covariance
    rf    : float — taux sans risque mensuel (log-rendement)

    Retourne
    --------
    (w_tangent, sharpe_tangent) : np.ndarray (N,), float
    """
    N = len(mu)
    excess = mu - rf

    def neg_sharpe(w):
        ret = float(w @ excess)
        var = float(w @ Sigma @ w)
        if var < 1e-14:
            return 1e6
        return -ret / np.sqrt(var)

    constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    bounds      = [(0.0, 1.0)] * N

    # Points d'initialisation : équipondéré + 1 poids unitaire par facteur
    starts = [np.ones(N) / N] + [
        np.eye(N)[i] * 0.6 + np.ones(N) * 0.4 / N for i in range(N)
    ]

    best_sharpe = -np.inf
    best_w      = np.ones(N) / N

    for w0 in starts:
        try:
            res = minimize(
                fun=neg_sharpe,
                x0=w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 500},
            )
            if res.success:
                w   = res.x
                var = float(w @ Sigma @ w)
                if var > 1e-14:
                    sr = -float(res.fun)
                    if sr > best_sharpe:
                        best_sharpe = sr
                        best_w      = w.copy()
        except Exception:
            continue

    return best_w, best_sharpe


def _mv_130_30(
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float,
    n_points: int = None,  # conservé pour compatibilité, non utilisé
) -> tuple[np.ndarray, float]:
    """
    Portefeuille tangent 130/30 par maximisation directe du Sharpe ratio.

    Stratégie 130/30 (papier, note 11) :
        - Exposition longue totale : sum(l) = 1.30
        - Exposition courte totale : sum(s) = 0.30
        - Exposition nette         : sum(w) = sum(l) - sum(s) = 1.00

    Reformulation : variables x = [l, s] avec w = l - s, l >= 0, s >= 0.

    Problème :
        max_x  ((l-s)' mu - rf) / sqrt((l-s)' Sigma (l-s))
        s.t.   sum(l) = 1.30,  sum(s) = 0.30,  l >= 0,  s >= 0

    Paramètres
    ----------
    mu    : np.ndarray (N,)  — rendements attendus
    Sigma : np.ndarray (N,N) — matrice de covariance
    rf    : float            — taux sans risque mensuel

    Retourne
    --------
    (w_tangent, sharpe_tangent) : np.ndarray (N,), float
    """
    N      = len(mu)
    excess = mu - rf

    def neg_sharpe_130(x):
        w   = x[:N] - x[N:]
        ret = float(w @ excess)
        var = float(w @ Sigma @ w)
        if var < 1e-14:
            return 1e6
        return -ret / np.sqrt(var)

    constraints = [
        {"type": "eq", "fun": lambda x: float(x[:N].sum() - LONG_TARGET)},
        {"type": "eq", "fun": lambda x: float(x[N:].sum() - SHORT_TARGET)},
    ]
    bounds = [(0.0, None)] * (2 * N)

    # Points d'initialisation : équipondéré + concentrations
    x0_base = np.concatenate([
        np.full(N, LONG_TARGET / N),
        np.full(N, SHORT_TARGET / N),
    ])
    starts = [x0_base]
    for i in range(N):
        x0 = x0_base.copy()
        x0[i]     = LONG_TARGET * 0.5
        x0[:N]   /= x0[:N].sum() / LONG_TARGET
        x0[N + i] = SHORT_TARGET * 0.5
        x0[N:]   /= x0[N:].sum() / SHORT_TARGET
        starts.append(x0)

    best_sharpe = -np.inf
    best_w      = np.full(N, LONG_TARGET / N) - np.full(N, SHORT_TARGET / N)

    for x0 in starts:
        try:
            res = minimize(
                fun=neg_sharpe_130,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 500},
            )
            if res.success:
                l, s = res.x[:N], res.x[N:]
                w    = l - s
                var  = float(w @ Sigma @ w)
                if var > 1e-14:
                    sr = -float(res.fun)
                    if sr > best_sharpe:
                        best_sharpe = sr
                        best_w      = w.copy()
        except Exception:
            continue

    return best_w, best_sharpe


# ==============================================================================
# SECTION 5 : BENCHMARK 1/N
# ==============================================================================

def _compute_benchmark_1N(
    monthly_oos: pd.DataFrame,
    rf_oos: pd.Series,
) -> dict:
    """
    Portefeuille équipondéré 1/N (Panel A du papier, Table 8).
    Poids constants : w_i = 1/5 pour chaque facteur.
    Rendement réalisé à chaque date : r_p,t = (1/5) * sum_i r_{i,t}
    """
    N = len(FACTEURS)
    w = np.ones(N) / N

    r_realized = monthly_oos[FACTEURS].values @ w  # (T,)
    rf_arr     = rf_oos.reindex(monthly_oos.index).fillna(0.0).values

    weights_df = pd.DataFrame(
        data=np.tile(w, (len(monthly_oos), 1)),
        index=monthly_oos.index,
        columns=FACTEURS,
    )
    returns_s = pd.Series(r_realized, index=monthly_oos.index, name="return")

    return {
        "weights": weights_df,
        "returns": returns_s,
    }


# ==============================================================================
# SECTION 6 : PIPELINE PRINCIPAL
# ==============================================================================

def run_mv_optimization(verbose: bool = True) -> tuple[dict, pd.DatetimeIndex, pd.DataFrame, pd.Series]:
    """
    Exécute la Section 5.1 complète : optimisation M-V pour toutes les
    combinaisons (prévisions × covariance) et les deux structures de portefeuille
    (long-only et 130/30).

    Retourne
    --------
    all_results : dict — clé = nom de stratégie, valeur = {"weights", "returns"}
    oos_dates   : pd.DatetimeIndex — 211 dates OOS (2000-01-31 à 2017-07-31)
    monthly_oos : pd.DataFrame — rendements réalisés OOS
    rf_oos      : pd.Series — taux sans risque OOS
    """
    if verbose:
        print("=" * 70)
        print("SECTION 5.1 — OPTIMISATION MOYENNE-VARIANCE (M-V)")
        print("Zhao et al. (2019), Panel B (long-only) et Panel C (130/30)")
        print("=" * 70)

    # ── 1. Chargement ──────────────────────────────────────────────────────

    if verbose:
        print("\n[1/5] Chargement des données...")

    monthly   = _charger_rendements()
    rf_series = _charger_rf()

    # Période OOS : 2000-01-31 à 2017-07-31
    masque_oos  = (monthly.index >= OOS_START) & (monthly.index <= OOS_END)
    monthly_oos = monthly.loc[masque_oos].copy()
    rf_oos      = rf_series.reindex(monthly_oos.index).fillna(0.0)
    oos_dates   = monthly_oos.index

    if verbose:
        print(f"  Période OOS : {oos_dates[0].date()} — {oos_dates[-1].date()} ({len(oos_dates)} mois)")

    # Prévisions de rendements
    rw    = _charger_previsions_rw(monthly)
    scsvr = _charger_previsions(CHEMIN_SC_SVR)
    dma   = _charger_previsions(CHEMIN_DMA)

    # Matrices de corrélation (début de mois → fin de mois)
    dcc  = _align_corr_to_eom(_charger_correlations(CHEMIN_DCC))
    adcc = _align_corr_to_eom(_charger_correlations(CHEMIN_ADCC))
    gas  = _align_corr_to_eom(_charger_correlations(CHEMIN_GAS))

    # Restriction à la période OOS
    rw    = rw.reindex(oos_dates)
    scsvr = scsvr.reindex(oos_dates)
    dma   = dma.reindex(oos_dates)
    dcc   = dcc.reindex(oos_dates)
    adcc  = adcc.reindex(oos_dates)
    gas   = gas.reindex(oos_dates)

    if verbose:
        print(f"  RW complet    : {rw.notna().all(axis=1).sum()} / {len(oos_dates)} mois")
        print(f"  SC-SVR complet: {scsvr.notna().all(axis=1).sum()} / {len(oos_dates)} mois")
        print(f"  DMA complet   : {dma.notna().all(axis=1).sum()} / {len(oos_dates)} mois")
        print(f"  DCC complet   : {dcc.notna().all(axis=1).sum()} / {len(oos_dates)} mois")
        print(f"  ADCC complet  : {adcc.notna().all(axis=1).sum()} / {len(oos_dates)} mois")
        print(f"  GAS complet   : {gas.notna().all(axis=1).sum()} / {len(oos_dates)} mois")

    # ── 2. Volatilités GARCH rolling ───────────────────────────────────────

    if verbose:
        print("\n[2/5] Calcul des volatilités conditionnelles GARCH(1,1) rolling...")

    sigma_t = _compute_garch_vol_rolling(monthly, oos_dates, verbose=verbose)

    if verbose:
        print(f"  Sigma complètes: {sigma_t.notna().all(axis=1).sum()} / {len(oos_dates)} mois")
        print("  Volatilités moyennes par facteur (décimal/mois) :")
        for f in FACTEURS:
            print(f"    {f}: {sigma_t[f].mean():.4f}")

    # ── 3. Benchmark 1/N ───────────────────────────────────────────────────

    if verbose:
        print("\n[3/5] Benchmark 1/N (Panel A du papier)...")

    all_results = {}
    all_results["1/N"] = _compute_benchmark_1N(monthly_oos, rf_oos)

    # ── 4. Optimisation MV par combinaison ─────────────────────────────────

    modeles_previsions = {"RW": rw, "SC-SVR": scsvr, "DMA": dma}
    modeles_cov        = {"DCC": dcc, "ADCC": adcc, "GAS": gas}

    combos = [
        (p_name, c_name)
        for p_name in modeles_previsions
        for c_name in modeles_cov
    ]

    if verbose:
        print(f"\n[4/5] Optimisation M-V — {len(combos)} combinaisons × 2 structures...")

    for (prev_name, cov_name) in tqdm(combos, desc="Combinaisons M-V"):
        prev_df = modeles_previsions[prev_name]
        corr_df = modeles_cov[cov_name]

        # Clés de résultats (convention papier, Table 8)
        key_lo    = f"{prev_name}-{cov_name}"          # Panel B (long-only)
        key_130   = f"{prev_name}-{cov_name}-S"        # Panel C (130/30, "-S" = short-selling)

        # Initialisation des séries temporelles de poids et rendements
        w_lo   = pd.DataFrame(index=oos_dates, columns=FACTEURS, dtype=float)
        w_130  = pd.DataFrame(index=oos_dates, columns=FACTEURS, dtype=float)
        r_lo   = pd.Series(np.nan, index=oos_dates, name="return")
        r_130  = pd.Series(np.nan, index=oos_dates, name="return")

        for date_t in oos_dates:
            # ── Données à la date t ──
            mu_vec  = prev_df.loc[date_t, FACTEURS].values.astype(float)
            sig_vec = sigma_t.loc[date_t, FACTEURS].values.astype(float)
            rf_t    = float(rf_oos.loc[date_t]) if date_t in rf_oos.index else 0.0

            # Vérifier la disponibilité des données
            if np.any(np.isnan(mu_vec)) or np.any(np.isnan(sig_vec)):
                continue
            if np.any(sig_vec <= 0):
                continue

            # Vérifier que la ligne de corrélation existe
            if date_t not in corr_df.index:
                continue
            corr_row = corr_df.loc[date_t]
            if corr_row.isna().all():
                continue

            # ── Construction de Sigma_t ──
            R_t     = _corr_row_to_matrix(corr_row)
            Sigma_t = _build_sigma(R_t, sig_vec)

            # Rendement réalisé à la date t
            r_realise = monthly_oos.loc[date_t, FACTEURS].values.astype(float)

            # ── Portefeuille long-only (Panel B) ──
            w_opt_lo, _ = _mv_longonly(mu_vec, Sigma_t, rf_t)
            w_lo.loc[date_t]  = w_opt_lo
            r_lo.loc[date_t]  = float(r_realise @ w_opt_lo)

            # ── Portefeuille 130/30 (Panel C) ──
            w_opt_130, _ = _mv_130_30(mu_vec, Sigma_t, rf_t)
            w_130.loc[date_t] = w_opt_130
            r_130.loc[date_t] = float(r_realise @ w_opt_130)

        all_results[key_lo]  = {"weights": w_lo,  "returns": r_lo}
        all_results[key_130] = {"weights": w_130, "returns": r_130}

    if verbose:
        print(f"\n[5/5] Optimisation terminée — {len(all_results)} stratégies calculées.")

    return all_results, oos_dates, monthly_oos, rf_oos


# ==============================================================================
# SECTION 7 : SAUVEGARDE DES RESULTATS
# ==============================================================================

def _compute_metrics(
    returns: pd.Series,
    rf: pd.Series,
    weights: pd.DataFrame,
) -> dict:
    """
    Calcule les métriques de performance d'un portefeuille.

    Métriques (Table 8 du papier) :
        - Annualized return (%)    : expm1(mean_monthly * 12) * 100
        - Sharpe ratio (annualisé) : mean(excess) / std(excess) * sqrt(12)
        - Sortino ratio (annualisé): mean(excess) / std(excess_negatif) * sqrt(12)
        - MDD (%)                  : Maximum Drawdown sur la valeur cumulée
        - Weights (mean)           : poids moyens par facteur sur la période

    Paramètres
    ----------
    returns : pd.Series — rendements mensuels réalisés du portefeuille
    rf      : pd.Series — taux sans risque mensuel
    weights : pd.DataFrame — poids du portefeuille par date

    Retourne
    --------
    dict des métriques
    """
    r = returns.dropna()
    if len(r) < 2:
        return {}

    rf_aligned = rf.reindex(r.index).fillna(0.0)
    excess     = r - rf_aligned

    # Rendement annualisé (log-rendement -> rendement simple)
    mean_monthly   = float(r.mean())
    ann_return_pct = float(np.expm1(mean_monthly * 12)) * 100.0

    # Sharpe ratio annualisé
    std_excess = float(excess.std(ddof=1))
    sharpe     = float(excess.mean() / std_excess) * np.sqrt(12) if std_excess > 0 else np.nan

    # Sortino ratio annualisé (dénominateur = volatilité des excès négatifs)
    neg_excess  = excess[excess < 0]
    std_neg     = float(neg_excess.std(ddof=1)) if len(neg_excess) > 1 else np.nan
    sortino     = float(excess.mean() / std_neg) * np.sqrt(12) if (std_neg and std_neg > 0) else np.nan

    # Maximum Drawdown (sur la valeur de portefeuille cumulée)
    cum_value = np.exp(r.cumsum())
    roll_max  = cum_value.cummax()
    drawdowns = (cum_value - roll_max) / roll_max
    mdd_pct   = float(drawdowns.min()) * 100.0

    # Poids moyens sur la période OOS
    mean_weights = weights.reindex(r.index).mean()

    return {
        "ann_return_pct": ann_return_pct,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "mdd_pct":        mdd_pct,
        "n_obs":          len(r),
        "mean_weights":   mean_weights.to_dict(),
    }


def save_results(
    all_results: dict,
    oos_dates: pd.DatetimeIndex,
    monthly_oos: pd.DataFrame,
    rf_oos: pd.Series,
    verbose: bool = True,
) -> None:
    """
    Sauvegarde les résultats d'optimisation M-V.

    Structure de sortie :
        results/optimization/
            weights_<stratégie>.csv      — poids par date et par facteur
            returns_<stratégie>.csv      — rendements réalisés par date
            summary_mv_optimization.csv  — tableau récapitulatif des métriques

        results/optimization/best_portfolio_tangent/
            tangent_<stratégie>.csv      — poids du portefeuille tangent moyen
                                           (poids moyens sur la période OOS)
    """
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    DIR_BEST.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for key, result in all_results.items():
        weights = result.get("weights")
        returns = result.get("returns")

        if weights is None or returns is None:
            continue

        # Sauvegarde des poids et rendements
        safe_key = key.replace("/", "_").replace(" ", "_")

        if isinstance(weights, pd.DataFrame):
            weights.to_csv(DIR_RESULTS / f"weights_{safe_key}.csv", date_format="%Y-%m-%d")
        else:
            # weights est un pd.Series (cas 1/N avec poids constants)
            weights.to_csv(DIR_RESULTS / f"weights_{safe_key}.csv")

        if isinstance(returns, pd.Series):
            returns.to_csv(DIR_RESULTS / f"returns_{safe_key}.csv", date_format="%Y-%m-%d")

        # Calcul des métriques
        if isinstance(weights, pd.DataFrame):
            metrics = _compute_metrics(returns, rf_oos, weights)
        else:
            w_df = pd.DataFrame(
                data=np.tile(weights.values, (len(oos_dates), 1)),
                index=oos_dates,
                columns=FACTEURS,
            )
            metrics = _compute_metrics(returns, rf_oos, w_df)

        if metrics:
            row = {"stratégie": key}
            row.update({k: v for k, v in metrics.items() if k != "mean_weights"})
            summary_rows.append(row)

            # Sauvegarde du portefeuille tangent (poids moyens)
            if "mean_weights" in metrics:
                tangent_s = pd.Series(metrics["mean_weights"], name=key)
                tangent_s.to_csv(DIR_BEST / f"tangent_{safe_key}.csv", header=True)

        if verbose:
            ann_r = metrics.get("ann_return_pct", np.nan)
            sr    = metrics.get("sharpe", np.nan)
            mdd   = metrics.get("mdd_pct", np.nan)
            print(f"  {key:30s} | ret={ann_r:+7.2f}%  SR={sr:+6.3f}  MDD={mdd:+7.2f}%")

    # Tableau récapitulatif
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).set_index("stratégie")
        df_summary.to_csv(DIR_RESULTS / "summary_mv_optimization.csv")
        if verbose:
            print(f"\n  Résumé sauvegardé : {DIR_RESULTS / 'summary_mv_optimization.csv'}")

    if verbose:
        print(f"  Dossier principal : {DIR_RESULTS}")
        print(f"  Portefeuilles tangents : {DIR_BEST}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    results, oos_dates, monthly_oos, rf_oos = run_mv_optimization(verbose=True)

    print("\n" + "=" * 70)
    print("SAUVEGARDE DES RESULTATS")
    print("=" * 70 + "\n")

    save_results(results, oos_dates, monthly_oos, rf_oos, verbose=True)

    print("\n" + "=" * 70)
    print("SECTION 5.1 TERMINEE")
    print("=" * 70)
