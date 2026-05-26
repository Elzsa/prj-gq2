# src/02_forecasting/individual_models/linear.py

import sys
from pathlib import Path

# ajout de la racine du projet au sys.path pour permettre les imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA

from config.splits import TRAIN_START, TRAIN_END, TEST_START, TEST_END

# noms des cinq facteurs Fama-French utilises dans le papier
FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]

# chemin vers les log-rendements produits par preprocess.py
CHEMIN_LOG_RETURNS = Path(__file__).resolve().parents[3] / "data" / "monthly_log_returns.csv"

# chemin de sortie des previsions individuelles lineaires
CHEMIN_SORTIE = Path(__file__).resolve().parents[3] / "data" / "02_forecasting" / "individual_predictions_linear.csv"

# nombre de jobs paralleles : -1 = tous les cores disponibles
N_JOBS = -1


# ==============================================================================
# SECTION 1 : CHARGEMENT DES DONNEES
# ==============================================================================

def charger_log_rendements(verbose: bool = True) -> pd.DataFrame:
    """Charge les log-rendements mensuels depuis data/monthly_log_returns.csv."""
    df = pd.read_csv(
        filepath_or_buffer=CHEMIN_LOG_RETURNS,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d"
    )
    df = df[FACTEURS]

    if verbose:
        print(f"Log-rendements charges : {df.shape[0]} observations, periode {df.index.min().date()} - {df.index.max().date()}")

    return df


# ==============================================================================
# SECTION 2 : MODELES SMA (Simple Moving Average)
# ==============================================================================
# Papier Appendice A Table A.1 (page 20) :
#   E(R_t) = (R_t-1 + ... + R_t-q) / q,  q = 3, ..., 30  ->  28 modeles
# SMA et EMA sont vectorises : pas de parallelisme necessaire.

def generer_previsions_sma(serie: pd.Series, periode_debut: str, periode_fin: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 28 previsions SMA (q de 3 a 30) pour une serie de rendements."""
    resultats = {}

    for q in range(3, 31):  # q = 3, 4, ..., 30  ->  28 modeles
        # shift(1) : prevision pour t utilise R_t-1, ..., R_t-q (pas de look-ahead bias)
        previsions = serie.shift(1).rolling(window=q).mean()
        masque = (previsions.index >= periode_debut) & (previsions.index <= periode_fin)
        resultats[f"SMA({q})"] = previsions.loc[masque]

    df_sma = pd.DataFrame(data=resultats)

    if verbose:
        print(f"SMA : {df_sma.shape[1]} modeles, {df_sma.shape[0]} observations sur {periode_debut} - {periode_fin}")

    return df_sma


# ==============================================================================
# SECTION 3 : MODELES EMA (Exponential Moving Average)
# ==============================================================================
# Papier Appendice A Table A.1 (page 20) :
#   E(R_t) = somme ponderee avec alpha' = 2/(1 + Ndays), q' = 3, ..., 30  ->  28 modeles

def generer_previsions_ema(serie: pd.Series, periode_debut: str, periode_fin: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 28 previsions EMA (q de 3 a 30) pour une serie de rendements."""
    resultats = {}

    for q in range(3, 31):  # q = 3, 4, ..., 30  ->  28 modeles
        # ewm(span=q) : alpha = 2/(1+q), adjust=False : recursif, shift(1) : pas de look-ahead bias
        previsions = serie.shift(1).ewm(span=q, adjust=False).mean()
        masque = (previsions.index >= periode_debut) & (previsions.index <= periode_fin)
        resultats[f"EMA({q})"] = previsions.loc[masque]

    df_ema = pd.DataFrame(data=resultats)

    if verbose:
        print(f"EMA : {df_ema.shape[1]} modeles, {df_ema.shape[0]} observations sur {periode_debut} - {periode_fin}")

    return df_ema


# ==============================================================================
# SECTION 4 : MODELES AR (AutoRegressive)
# ==============================================================================
# Papier Appendice A Table A.1 (page 20) :
#   E(R_t) = beta_0 + somme_i beta_i * R_t-i,  q = 1, ..., 24  ->  24 modeles
#
# Strategie : params estimes sur TRAIN, apply(refit=False) sur TRAIN+TEST.
# fittedvalues[t] = E[y_t | y_1,...,y_{t-1}, theta_fixe] -> vrais 1-step ahead forecasts.
# Gain vs expanding window : ~200x (0.2s vs 43s par modele).
#
# Parallelisme niveau 1 : les 24 ordres AR sont independants -> Parallel sur q.

def _prevoir_ar_un_ordre(q: int, serie: pd.Series, serie_train: pd.Series, periode_pred_debut: str, periode_pred_fin: str) -> tuple:
    """Estime AR(q) sur serie_train et produit les previsions 1-step via apply(refit=False). Retourne (nom, serie)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modele         = ARIMA(endog=serie_train, order=(q, 0, 0), trend="c")
        res_train      = modele.fit(method="innovations_mle", low_memory=True)
        res_full       = res_train.apply(endog=serie, refit=False)
        previsions     = res_full.fittedvalues.loc[periode_pred_debut:periode_pred_fin]

    previsions.name = f"AR({q})"
    return (f"AR({q})", previsions)


def generer_previsions_ar(serie: pd.Series, serie_train: pd.Series, periode_pred_debut: str, periode_pred_fin: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 24 previsions AR en parallele (q de 1 a 24) pour une serie de rendements."""
    if verbose:
        print(f"  AR : 24 estimations en parallele (N_JOBS={N_JOBS})...")

    resultats_liste = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_prevoir_ar_un_ordre)(
            q=q,
            serie=serie,
            serie_train=serie_train,
            periode_pred_debut=periode_pred_debut,
            periode_pred_fin=periode_pred_fin
        )
        for q in range(1, 25)  # q = 1, ..., 24
    )

    # reconstruction dans l'ordre AR(1), AR(2), ...
    resultats = dict(resultats_liste)
    df_ar = pd.DataFrame(data={f"AR({q})": resultats[f"AR({q})"] for q in range(1, 25)})

    if verbose:
        print(f"AR : {df_ar.shape[1]} modeles, {df_ar.shape[0]} observations sur {periode_pred_debut} - {periode_pred_fin}")

    return df_ar


# ==============================================================================
# SECTION 5 : MODELES ARMA (AutoRegressive Moving Average)
# ==============================================================================
# Papier Appendice A Table A.1 (page 20) :
#   E(R_t) = phi_0 + somme_j phi_j * R_t-j + a_0 + somme_k w_k * a_t-k
#   m_prime dans [1,15], n_prime dans [1,14]  ->  15 x 14 = 210 modeles
#
# Ambiguite du papier : 15x15=225 mais le papier annonce 210.
# Decision retenue : n_prime dans [1,14] (seule decomposition entiere donnant 210).
#
# Parallelisme niveau 1 : les 210 combinaisons (p,q) sont independantes -> Parallel.

def _prevoir_arma_un_ordre(p: int, q: int, serie: pd.Series, serie_train: pd.Series, periode_pred_debut: str, periode_pred_fin: str) -> tuple:
    """Estime ARMA(p,q) sur serie_train et produit les previsions 1-step via apply(refit=False). Retourne (nom, serie)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modele     = ARIMA(endog=serie_train, order=(p, 0, q), trend="c")
            res_train  = modele.fit(method="innovations_mle", low_memory=True)
            res_full   = res_train.apply(endog=serie, refit=False)
            previsions = res_full.fittedvalues.loc[periode_pred_debut:periode_pred_fin]
    except Exception:
        index_pred = serie.loc[periode_pred_debut:periode_pred_fin].index
        previsions = pd.Series(data=np.nan, index=index_pred)

    previsions.name = f"ARMA({p},{q})"
    return (f"ARMA({p},{q})", previsions)


def generer_previsions_arma(serie: pd.Series, serie_train: pd.Series, periode_pred_debut: str, periode_pred_fin: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 210 previsions ARMA en parallele (m' de 1 a 15, n' de 1 a 14)."""
    combinaisons = [(p, q) for p in range(1, 16) for q in range(1, 15)]  # 15 x 14 = 210

    if verbose:
        print(f"  ARMA : {len(combinaisons)} estimations en parallele (N_JOBS={N_JOBS})...")

    resultats_liste = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_prevoir_arma_un_ordre)(
            p=p,
            q=q,
            serie=serie,
            serie_train=serie_train,
            periode_pred_debut=periode_pred_debut,
            periode_pred_fin=periode_pred_fin
        )
        for p, q in combinaisons
    )

    resultats = dict(resultats_liste)
    df_arma = pd.DataFrame(data={f"ARMA({p},{q})": resultats[f"ARMA({p},{q})"] for p, q in combinaisons})

    if verbose:
        print(f"ARMA : {df_arma.shape[1]} modeles, {df_arma.shape[0]} observations sur {periode_pred_debut} - {periode_pred_fin}")

    return df_arma


# ==============================================================================
# SECTION 6 : PIPELINE POUR UN FACTEUR
# ==============================================================================

def generer_previsions_lineaires_facteur(serie: pd.Series, nom_facteur: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 290 previsions lineaires pour un facteur sur la periode TEST (1984-1999)."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Facteur : {nom_facteur}")
        print(f"{'='*60}")

    serie_train    = serie.loc[TRAIN_START:TRAIN_END]   # 1965-1983 : estimation des params
    serie_insample = serie.loc[TRAIN_START:TEST_END]    # 1965-1999 : apply() et SMA/EMA

    if verbose:
        print("\nSMA (28 modeles)...")
    df_sma = generer_previsions_sma(serie=serie_insample, periode_debut=TEST_START, periode_fin=TEST_END, verbose=verbose)

    if verbose:
        print("\nEMA (28 modeles)...")
    df_ema = generer_previsions_ema(serie=serie_insample, periode_debut=TEST_START, periode_fin=TEST_END, verbose=verbose)

    if verbose:
        print("\nAR (24 modeles)...")
    df_ar = generer_previsions_ar(serie=serie_insample, serie_train=serie_train, periode_pred_debut=TEST_START, periode_pred_fin=TEST_END, verbose=verbose)

    if verbose:
        print("\nARMA (210 modeles)...")
    df_arma = generer_previsions_arma(serie=serie_insample, serie_train=serie_train, periode_pred_debut=TEST_START, periode_pred_fin=TEST_END, verbose=verbose)

    # concatenation : 28 + 28 + 24 + 210 = 290 colonnes
    df_complet = pd.concat(objs=[df_sma, df_ema, df_ar, df_arma], axis=1)

    assert df_complet.shape[1] == 290, f"Attendu 290 modeles, obtenu {df_complet.shape[1]}"

    if verbose:
        n_nans  = df_complet.isna().sum().sum()
        n_total = df_complet.shape[1] * df_complet.shape[0]
        print(f"\nTotal : {df_complet.shape[1]} modeles, {df_complet.shape[0]} observations")
        print(f"Valeurs manquantes : {n_nans} sur {n_total} ({100*n_nans/n_total:.1f}%)")

    return df_complet


# ==============================================================================
# SECTION 7 : PIPELINE COMPLET POUR LES 5 FACTEURS (parallelisme niveau 2)
# ==============================================================================

def _generer_facteur_wrapper(facteur: str, df_log: pd.DataFrame, verbose: bool) -> tuple:
    """Wrapper pour generer_previsions_lineaires_facteur en parallele. Retourne (facteur, DataFrame)."""
    return (facteur, generer_previsions_lineaires_facteur(serie=df_log[facteur], nom_facteur=facteur, verbose=verbose))


def executer_previsions_lineaires(verbose: bool = True) -> dict:
    """Execute les 290 previsions lineaires pour les 5 facteurs en parallele et sauvegarde en CSV multi-index."""
    print("ETAPE 02_FORECASTING INDIVIDUAL LINEAR ===================") if verbose else None

    df_log = charger_log_rendements(verbose=verbose)

    if verbose:
        print(f"\nParallelisme niveau 2 : 5 facteurs en parallele (N_JOBS={N_JOBS})...")

    # parallelisme niveau 2 : les 5 facteurs sont independants
    # prefer="processes" : taches longues et CPU-bound (ARIMA), le fork est amorti
    resultats_liste = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(_generer_facteur_wrapper)(
            facteur=facteur,
            df_log=df_log,
            verbose=verbose
        )
        for facteur in FACTEURS
    )

    previsions_par_facteur = dict(resultats_liste)

    # sauvegarde CSV avec multi-index (facteur, modele) en colonnes
    df_global = pd.concat(objs=previsions_par_facteur, axis=1)
    df_global.columns.names = ["facteur", "modele"]

    CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    df_global.to_csv(path_or_buf=CHEMIN_SORTIE, date_format="%Y-%m-%d")

    if verbose:
        print(f"\nPrevisions sauvegardees : {CHEMIN_SORTIE}")
        print(f"Dimensions : {df_global.shape[0]} dates x {df_global.shape[1]} colonnes (5 facteurs x 290 modeles)")
        print("ETAPE 02_FORECASTING INDIVIDUAL LINEAR END ===============")

    return previsions_par_facteur


# ==============================================================================
# SECTION 8 : CHARGEMENT
# ==============================================================================

def charger_previsions_lineaires(verbose: bool = True) -> dict:
    """Charge les previsions lineaires sauvegardees depuis le CSV multi-index."""
    df_global = pd.read_csv(
        filepath_or_buffer=CHEMIN_SORTIE,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
        header=[0, 1]
    )

    previsions_par_facteur = {}
    for facteur in FACTEURS:
        previsions_par_facteur[facteur] = df_global[facteur]

    if verbose:
        print(f"Previsions lineaires chargees : {df_global.shape[0]} dates, {len(FACTEURS)} facteurs, {df_global.shape[1] // len(FACTEURS)} modeles par facteur")

    return previsions_par_facteur


if __name__ == "__main__":
    previsions = executer_previsions_lineaires(verbose=True)
    a = True # début 13h02