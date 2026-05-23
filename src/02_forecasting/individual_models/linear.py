# src/02_forecasting/individual_models/linear.py

import sys
from pathlib import Path

# ajout de la racine du projet au sys.path pour permettre les imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from config.splits import TRAIN_START, TRAIN_END, TEST_START, TEST_END

# noms des cinq facteurs Fama-French utilisés dans le papier
FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]

# chemin vers les log-rendements produits par preprocess.py
CHEMIN_LOG_RETURNS = Path(__file__).resolve().parents[3] / "data" / "monthly_log_returns.csv"

# chemin de sortie des prévisions individuelles linéaires
CHEMIN_SORTIE = Path(__file__).resolve().parents[3] / "data" / "02_forecasting" / "individual_predictions_linear.csv"


# ==============================================================================
# SECTION 1 : CHARGEMENT DES DONNÉES
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
# SECTION 2 : MODÈLES SMA (Simple Moving Average)
# ==============================================================================
# Papier Appendice A Table A.1 (page 20) :
#   E(R_t) = (R_t-1 + ... + R_t-q) / q,  q = 3, ..., 30  ->  28 modeles

def prevoir_sma(serie: pd.Series, q: int, periode_debut: str, periode_fin: str, verbose: bool = True) -> pd.Series:
    """Calcule les previsions 1 pas en avant par moyenne mobile simple d'ordre q sur la periode [debut, fin]."""
    # la prevision a t est la moyenne des q rendements passes : R_t-1, ..., R_t-q
    # shift(1) decale d'une periode pour eviter le look-ahead bias
    previsions = serie.shift(1).rolling(window=q).mean()

    masque = (previsions.index >= periode_debut) & (previsions.index <= periode_fin)
    previsions = previsions.loc[masque]

    if verbose:
        n_valides = previsions.notna().sum()
        print(f"  SMA({q}) : {n_valides} previsions valides sur {len(previsions)} attendues")

    return previsions


def generer_previsions_sma(serie: pd.Series, periode_debut: str, periode_fin: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 28 previsions SMA (q de 3 a 30) pour une serie de rendements."""
    resultats = {}

    for q in range(3, 31):  # q = 3, 4, ..., 30  ->  28 modeles
        nom_modele = f"SMA({q})"
        resultats[nom_modele] = prevoir_sma(serie=serie, q=q, periode_debut=periode_debut, periode_fin=periode_fin, verbose=False)

    df_sma = pd.DataFrame(data=resultats)

    if verbose:
        print(f"SMA : {df_sma.shape[1]} modeles, {df_sma.shape[0]} observations sur {periode_debut} - {periode_fin}")

    return df_sma


# ==============================================================================
# SECTION 3 : MODÈLES EMA (Exponential Moving Average)
# ==============================================================================
# Papier Appendice A Table A.1 (page 20) :
#   E(R_t) = somme ponderee de R_t-1, ..., R_t-q' avec alpha' = 2/(1 + Ndays)
#   Ndays est le nombre de jours de trading et q' = 3, ..., 30  ->  28 modeles
# Le papier definit Ndays = q' car les donnees sont mensuelles.
# alpha = 2 / (1 + q') est le facteur de lissage standard de l'EMA.

def prevoir_ema(serie: pd.Series, q: int, periode_debut: str, periode_fin: str, verbose: bool = True) -> pd.Series:
    """Calcule les previsions 1 pas en avant par moyenne mobile exponentielle d'ordre q sur la periode [debut, fin]."""
    alpha = 2.0 / (1.0 + q)

    # pandas ewm(span=q) utilise exactement alpha = 2/(1+span), ce qui correspond a la formule du papier
    # adjust=False : implementation recursive R_t_ema = alpha * R_t + (1 - alpha) * R_t-1_ema
    # shift(1) : la prevision pour t utilise les donnees jusqu'en t-1 (pas de look-ahead bias)
    previsions = serie.shift(1).ewm(span=q, adjust=False).mean()

    masque = (previsions.index >= periode_debut) & (previsions.index <= periode_fin)
    previsions = previsions.loc[masque]

    if verbose:
        n_valides = previsions.notna().sum()
        print(f"  EMA({q}) alpha={alpha:.4f} : {n_valides} previsions valides sur {len(previsions)} attendues")

    return previsions


def generer_previsions_ema(serie: pd.Series, periode_debut: str, periode_fin: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 28 previsions EMA (q de 3 a 30) pour une serie de rendements."""
    resultats = {}

    for q in range(3, 31):  # q = 3, 4, ..., 30  ->  28 modeles
        nom_modele = f"EMA({q})"
        resultats[nom_modele] = prevoir_ema(serie=serie, q=q, periode_debut=periode_debut, periode_fin=periode_fin, verbose=False)

    df_ema = pd.DataFrame(data=resultats)

    if verbose:
        print(f"EMA : {df_ema.shape[1]} modeles, {df_ema.shape[0]} observations sur {periode_debut} - {periode_fin}")

    return df_ema


# ==============================================================================
# SECTION 4 : MODÈLES AR (AutoRegressive)
# ==============================================================================
# Papier Appendice A Table A.1 (page 20) :
#   E(R_t) = beta_0 + somme_i beta_i * R_t-i,  q = 1, ..., 24  ->  24 modeles
#
# Strategie d'estimation retenue : parametres estimes sur TRAIN (1965-1983),
# previsions 1-step rolling sur TEST (1984-1999) via apply(refit=False).
# Cela signifie que les coefficients du modele sont fixes et que seuls les
# lags observes (vraies valeurs passees) sont mis a jour a chaque pas.
#
# Alternative ecartee : expanding window (re-estimation a chaque date t).
# Raison : le papier ne mentionne pas explicitement l'expanding window pour
# les modeles individuels (uniquement pour le DMA, Section 3.2.3).
# De plus, l'expanding window multiplie le temps de calcul par 191
# sans gain methodologique justifie par le texte.
# Gain de performance : ~200x (0.2s vs 43s par modele).

def prevoir_ar(serie: pd.Series, q: int, serie_train: pd.Series, periode_pred_debut: str, periode_pred_fin: str, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par AR(q) : params estimes sur serie_train, rolling 1-step sur periode de prediction."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # estimation des parametres sur TRAIN uniquement
        modele = ARIMA(endog=serie_train, order=(q, 0, 0), trend="c")
        resultat_train = modele.fit(method="innovations_mle", low_memory=True)

        # apply(refit=False) : rejoue le filtre sur la serie complete avec les params fixes
        # fittedvalues[t] = E[y_t | y_1,...,y_{t-1}, theta_fixe] -> vrais 1-step ahead forecasts
        resultat_full = resultat_train.apply(endog=serie, refit=False)
        previsions = resultat_full.fittedvalues.loc[periode_pred_debut:periode_pred_fin]

    previsions.name = f"AR({q})"

    if verbose:
        n_valides = previsions.notna().sum()
        print(f"  AR({q}) : {n_valides}/{len(previsions)} previsions valides")

    return previsions


def generer_previsions_ar(serie: pd.Series, serie_train: pd.Series, periode_pred_debut: str, periode_pred_fin: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 24 previsions AR (q de 1 a 24) pour une serie de rendements."""
    resultats = {}

    for q in range(1, 25):  # q = 1, 2, ..., 24  ->  24 modeles
        if verbose:
            print(f"  Estimation AR({q})...")
        resultats[f"AR({q})"] = prevoir_ar(
            serie=serie,
            q=q,
            serie_train=serie_train,
            periode_pred_debut=periode_pred_debut,
            periode_pred_fin=periode_pred_fin,
            verbose=False
        )

    df_ar = pd.DataFrame(data=resultats)

    if verbose:
        print(f"AR : {df_ar.shape[1]} modeles, {df_ar.shape[0]} observations sur {periode_pred_debut} - {periode_pred_fin}")

    return df_ar


# ==============================================================================
# SECTION 5 : MODÈLES ARMA (AutoRegressive Moving Average)
# ==============================================================================
# Papier Appendice A Table A.1 (page 20) :
#   E(R_t) = phi_0 + somme_j phi_j * R_t-j + a_0 + somme_k w_k * a_t-k
#   m_prime, n_prime = 1, ..., 15 croises  ->  210 modeles
#
# Decompte : le papier annonce 210 modeles ARMA.
# 15 x 15 = 225, donc 15 combinaisons sont exclues.
# La seule facon d'obtenir 210 est m' dans [1,15] et n' dans [1,14] : 15 x 14 = 210.
# Ambiguite du papier : il n'est pas precise quelles combinaisons sont exclues.
# Decision retenue : n_prime va de 1 a 14 (hypothese la plus parcimonieuse).
#
# Meme strategie que AR : params estimes sur TRAIN, apply(refit=False) sur TEST.

def prevoir_arma(serie: pd.Series, p: int, q: int, serie_train: pd.Series, periode_pred_debut: str, periode_pred_fin: str, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par ARMA(p,q) : params estimes sur serie_train, rolling 1-step sur periode de prediction."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modele = ARIMA(endog=serie_train, order=(p, 0, q), trend="c")
            resultat_train = modele.fit(method="innovations_mle", low_memory=True)
            resultat_full = resultat_train.apply(endog=serie, refit=False)
            previsions = resultat_full.fittedvalues.loc[periode_pred_debut:periode_pred_fin]
    except Exception:
        # en cas d'echec de convergence : NaN sur toute la periode
        index_pred = serie.loc[periode_pred_debut:periode_pred_fin].index
        previsions = pd.Series(data=np.nan, index=index_pred)

    previsions.name = f"ARMA({p},{q})"

    if verbose:
        n_valides = previsions.notna().sum()
        print(f"  ARMA({p},{q}) : {n_valides}/{len(previsions)} previsions valides")

    return previsions


def generer_previsions_arma(serie: pd.Series, serie_train: pd.Series, periode_pred_debut: str, periode_pred_fin: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 210 previsions ARMA (m' de 1 a 15, n' de 1 a 14) pour une serie de rendements."""
    # m' dans [1,15] x n' dans [1,14] = 15 x 14 = 210 modeles
    resultats = {}

    for p in range(1, 16):      # m' = 1, ..., 15
        for q in range(1, 15):  # n' = 1, ..., 14
            nom_modele = f"ARMA({p},{q})"
            if verbose:
                print(f"  Estimation {nom_modele}...")
            resultats[nom_modele] = prevoir_arma(
                serie=serie,
                p=p,
                q=q,
                serie_train=serie_train,
                periode_pred_debut=periode_pred_debut,
                periode_pred_fin=periode_pred_fin,
                verbose=False
            )

    df_arma = pd.DataFrame(data=resultats)

    if verbose:
        print(f"ARMA : {df_arma.shape[1]} modeles, {df_arma.shape[0]} observations sur {periode_pred_debut} - {periode_pred_fin}")

    return df_arma


# ==============================================================================
# SECTION 6 : PIPELINE COMPLET POUR UN FACTEUR
# ==============================================================================

def generer_previsions_lineaires_facteur(serie: pd.Series, nom_facteur: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les 290 previsions lineaires pour un facteur sur la periode TEST (1984-1999)."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Facteur : {nom_facteur}")
        print(f"{'='*60}")

    # serie d'estimation : TRAIN uniquement (1965-1983)
    serie_train = serie.loc[TRAIN_START:TRAIN_END]

    # serie complete in-sample pour SMA/EMA et pour apply() des AR/ARMA
    serie_insample = serie.loc[TRAIN_START:TEST_END]

    # --- SMA : 28 modeles ---
    if verbose:
        print("\nSMA (28 modeles)...")
    df_sma = generer_previsions_sma(
        serie=serie_insample,
        periode_debut=TEST_START,
        periode_fin=TEST_END,
        verbose=verbose
    )

    # --- EMA : 28 modeles ---
    if verbose:
        print("\nEMA (28 modeles)...")
    df_ema = generer_previsions_ema(
        serie=serie_insample,
        periode_debut=TEST_START,
        periode_fin=TEST_END,
        verbose=verbose
    )

    # --- AR : 24 modeles ---
    if verbose:
        print("\nAR (24 modeles)...")
    df_ar = generer_previsions_ar(
        serie=serie_insample,
        serie_train=serie_train,
        periode_pred_debut=TEST_START,
        periode_pred_fin=TEST_END,
        verbose=verbose
    )

    # --- ARMA : 210 modeles ---
    if verbose:
        print("\nARMA (210 modeles)...")
    df_arma = generer_previsions_arma(
        serie=serie_insample,
        serie_train=serie_train,
        periode_pred_debut=TEST_START,
        periode_pred_fin=TEST_END,
        verbose=verbose
    )

    # concatenation horizontale des 4 familles : 28 + 28 + 24 + 210 = 290 colonnes
    df_complet = pd.concat(objs=[df_sma, df_ema, df_ar, df_arma], axis=1)

    assert df_complet.shape[1] == 290, f"Attendu 290 modeles, obtenu {df_complet.shape[1]}"

    if verbose:
        n_nans = df_complet.isna().sum().sum()
        n_total = df_complet.shape[1] * df_complet.shape[0]
        print(f"\nTotal : {df_complet.shape[1]} modeles, {df_complet.shape[0]} observations")
        print(f"Valeurs manquantes (NaN) : {n_nans} sur {n_total} ({100*n_nans/n_total:.1f}%)")

    return df_complet


# ==============================================================================
# SECTION 7 : PIPELINE COMPLET POUR LES 5 FACTEURS
# ==============================================================================

def executer_previsions_lineaires(verbose: bool = True) -> dict:
    """Execute les 290 previsions lineaires pour les 5 facteurs et sauvegarde en CSV multi-index."""
    print("ETAPE 02_FORECASTING INDIVIDUAL LINEAR ===================") if verbose else None

    df_log = charger_log_rendements(verbose=verbose)

    previsions_par_facteur = {}

    for facteur in FACTEURS:
        serie_facteur = df_log[facteur]
        df_previsions = generer_previsions_lineaires_facteur(
            serie=serie_facteur,
            nom_facteur=facteur,
            verbose=verbose
        )
        previsions_par_facteur[facteur] = df_previsions

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
# SECTION 8 : FONCTIONS UTILITAIRES DE CHARGEMENT
# ==============================================================================

def charger_previsions_lineaires(verbose: bool = True) -> dict:
    """Charge les previsions lineaires sauvegardees depuis le CSV multi-index."""
    df_global = pd.read_csv(
        filepath_or_buffer=CHEMIN_SORTIE,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
        header=[0, 1]  # multi-index de colonnes : (facteur, modele)
    )

    previsions_par_facteur = {}
    for facteur in FACTEURS:
        previsions_par_facteur[facteur] = df_global[facteur]

    if verbose:
        print(f"Previsions lineaires chargees : {df_global.shape[0]} dates, {len(FACTEURS)} facteurs, {df_global.shape[1] // len(FACTEURS)} modeles par facteur")

    return previsions_par_facteur


if __name__ == "__main__":
    previsions = executer_previsions_lineaires(verbose=True)
    a = True # 23h30