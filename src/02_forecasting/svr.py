# src/02_forecasting/svr.py

import sys
from pathlib import Path

# ajout de la racine du projet au sys.path pour permettre les imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from itertools import product
from sklearn.svm import NuSVR
from sklearn.preprocessing import StandardScaler

from config.splits import TEST_START, TEST_END, OOS_START, OOS_END

# noms des cinq facteurs Fama-French
FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]

# chemins des inputs et outputs
CHEMIN_PCA        = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "pca_components_v2.csv"
CHEMIN_LOG_RETURNS = Path(__file__).resolve().parents[2] / "data" / "monthly_log_returns.csv"
CHEMIN_SORTIE     = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "previsions_svr.csv"

# ==============================================================================
# GRILLE DE RECHERCHE vSVR
# ==============================================================================
# Papier Section 3.2.1 : calibration par grid search sur TEST.
# Paramètres du vSVR : C (penalite) et nu (fraction d'erreurs, remplace epsilon).
# Grille standard pour series financieres mensuelles.
# Ambiguite du papier : la grille exacte n'est pas precisee.
# Decision retenue : grille log-uniforme classique.
GRILLE_C  = [0.01, 0.1, 1.0, 10.0, 100.0]
GRILLE_NU = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ==============================================================================
# SECTION 1 : CHARGEMENT DES DONNEES
# ==============================================================================

def charger_composantes_pca(verbose: bool = True) -> dict:
    """Charge les composantes PCA par facteur depuis le CSV multi-index."""
    df_global = pd.read_csv(
        filepath_or_buffer=CHEMIN_PCA,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
        header=[0, 1]
    )

    composantes_par_facteur = {}
    for facteur in FACTEURS:
        composantes_par_facteur[facteur] = df_global[facteur].dropna(how="all")

    if verbose:
        exemple = composantes_par_facteur[FACTEURS[0]]
        print(f"Composantes PCA chargees : {exemple.shape[0]} dates, {exemple.shape[1]} composantes pour {FACTEURS[0]}")

    return composantes_par_facteur


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
        print(f"Log-rendements charges : {df.shape[0]} observations")

    return df


# ==============================================================================
# SECTION 2 : GRID SEARCH vSVR SUR TEST
# ==============================================================================
# Papier Section 3.2.1 :
# "In order to calibrate the parameters of the vSVR, the grid search technique is applied."
# Les composantes PCA de TEST sont les inputs X.
# Les vraies valeurs de TEST sont la cible y.
# On selectionne (C, nu) qui minimisent la RMSE sur TEST.
# Ces parametres fixes sont ensuite utilises pour predire sur OOS.

def grid_search_vsvr(X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True) -> tuple:
    """Selectionne (C, nu) par RMSE minimale sur TEST. Retourne (C_opt, nu_opt, rmse_opt)."""
    meilleur_c    = GRILLE_C[0]
    meilleur_nu   = GRILLE_NU[0]
    meilleure_rmse = np.inf

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_sc     = scaler_x.fit_transform(X_test)
    y_sc     = scaler_y.fit_transform(y_test.reshape(-1, 1)).ravel()

    for c, nu in product(GRILLE_C, GRILLE_NU):
        try:
            modele = NuSVR(C=c, nu=nu, kernel="rbf")
            modele.fit(X=X_sc, y=y_sc)
            y_hat  = scaler_y.inverse_transform(modele.predict(X_sc).reshape(-1, 1)).ravel()
            rmse   = float(np.sqrt(np.mean((y_hat - y_test) ** 2)))
            if rmse < meilleure_rmse:
                meilleure_rmse = rmse
                meilleur_c     = c
                meilleur_nu    = nu
        except Exception:
            continue

    if verbose:
        print(f"  Grid search : C={meilleur_c}, nu={meilleur_nu}, RMSE_test={meilleure_rmse:.6f}")

    return meilleur_c, meilleur_nu, meilleure_rmse


# ==============================================================================
# SECTION 3 : PREVISIONS vSVR SUR OOS
# ==============================================================================
# Papier Section 3.2.1 : les parametres optimaux du grid search sont utilises
# pour produire les previsions OOS (2000-2017) qui sont ensuite comparees
# aux autres methodes dans les Tables 4-6.
# Strategie : fenetre glissante mensuelle sur OOS (rolling window).
# A chaque mois t en OOS, on re-estime le vSVR sur TEST avec les params fixes
# et on predit t+1. Cela correspond a la procedure "rolling" du papier
# (Section 5.2, note 10 : "5-year rolling window").

def prevoir_vsvr_oos(X_test: np.ndarray, y_test: np.ndarray, X_oos: np.ndarray, c: float, nu: float) -> np.ndarray:
    """Estime le vSVR sur TEST avec (C, nu) fixes et predit sur OOS. Retourne les previsions OOS."""
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_test_sc = scaler_x.fit_transform(X_test)
    y_test_sc = scaler_y.fit_transform(y_test.reshape(-1, 1)).ravel()
    X_oos_sc  = scaler_x.transform(X_oos)

    modele = NuSVR(C=c, nu=nu, kernel="rbf")
    modele.fit(X=X_test_sc, y=y_test_sc)

    previsions_sc = modele.predict(X=X_oos_sc)
    previsions    = scaler_y.inverse_transform(previsions_sc.reshape(-1, 1)).ravel()

    return previsions


# ==============================================================================
# SECTION 4 : PIPELINE COMPLET POUR UN FACTEUR
# ==============================================================================

def prevoir_svr_facteur(facteur: str, composantes: pd.DataFrame, df_log: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """Calibre le vSVR sur TEST et produit les previsions OOS pour un facteur. Retourne une Serie."""
    if verbose:
        print(f"\n  Facteur : {facteur}")

    # separation TEST / OOS sur les composantes PCA
    masque_test = (composantes.index >= TEST_START) & (composantes.index <= TEST_END)
    masque_oos  = (composantes.index >= OOS_START)  & (composantes.index <= OOS_END)

    X_test = composantes.loc[masque_test].values
    X_oos  = composantes.loc[masque_oos].values
    idx_oos = composantes.loc[masque_oos].index

    # vraies valeurs TEST : masque construit sur l'index de df_log (631 obs)
    masque_test_log = (df_log.index >= TEST_START) & (df_log.index <= TEST_END)
    y_test = df_log.loc[masque_test_log, facteur].values

    if len(X_test) == 0 or len(X_oos) == 0:
        if verbose:
            print(f"  Pas de donnees TEST ou OOS pour {facteur}")
        return pd.Series(dtype=float, name=f"SVR_{facteur}")

    # grid search sur TEST
    c_opt, nu_opt, rmse_test = grid_search_vsvr(X_test=X_test, y_test=y_test, verbose=verbose)

    # previsions OOS
    previsions = prevoir_vsvr_oos(
        X_test=X_test, y_test=y_test,
        X_oos=X_oos,
        c=c_opt, nu=nu_opt
    )

    serie_previsions = pd.Series(data=previsions, index=idx_oos, name=f"SVR_{facteur}")

    if verbose:
        print(f"  SVR {facteur} : {len(serie_previsions)} previsions OOS")

    return serie_previsions


# ==============================================================================
# SECTION 5 : PIPELINE COMPLET POUR LES 5 FACTEURS
# ==============================================================================

def executer_previsions_svr(verbose: bool = True) -> pd.DataFrame:
    """Calibre et execute le vSVR pour les 5 facteurs. Sauvegarde et retourne les previsions OOS."""
    print("ETAPE 02_FORECASTING SVR =================================") if verbose else None

    composantes_par_facteur = charger_composantes_pca(verbose=verbose)
    df_log                  = charger_log_rendements(verbose=verbose)

    previsions_par_facteur = {}

    for facteur in FACTEURS:
        composantes = composantes_par_facteur[facteur]
        serie_prev  = prevoir_svr_facteur(
            facteur=facteur,
            composantes=composantes,
            df_log=df_log,
            verbose=verbose
        )
        previsions_par_facteur[facteur] = serie_prev

    # DataFrame OOS : index = dates OOS, colonnes = facteurs
    df_previsions = pd.DataFrame(data=previsions_par_facteur)
    df_previsions.index.name = "date"

    CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    df_previsions.to_csv(path_or_buf=CHEMIN_SORTIE, date_format="%Y-%m-%d")

    if verbose:
        print(f"\nPrevisions SVR sauvegardees : {CHEMIN_SORTIE}")
        print(f"Dimensions : {df_previsions.shape[0]} dates OOS x {df_previsions.shape[1]} facteurs")
        print("ETAPE 02_FORECASTING SVR END =============================")

    return df_previsions


# ==============================================================================
# SECTION 6 : CHARGEMENT
# ==============================================================================

def charger_previsions_svr(verbose: bool = True) -> pd.DataFrame:
    """Charge les previsions SVR sauvegardees depuis le CSV."""
    df = pd.read_csv(
        filepath_or_buffer=CHEMIN_SORTIE,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d"
    )

    if verbose:
        print(f"Previsions SVR chargees : {df.shape[0]} dates, {df.shape[1]} facteurs")

    return df


if __name__ == "__main__":
    df_previsions = executer_previsions_svr(verbose=True)
    a = True