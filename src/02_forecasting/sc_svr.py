# src/02_forecasting/sc_svr.py

import sys
from pathlib import Path

# ajout de la racine du projet au sys.path pour permettre les imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle
import numpy as np
import pandas as pd
from sklearn.svm import NuSVR
from sklearn.preprocessing import StandardScaler

from config.splits import TEST_START, TEST_END, OOS_START, OOS_END

# noms des cinq facteurs Fama-French
FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]

# chemins des inputs et outputs
CHEMIN_PCA           = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "pca_components.csv"
CHEMIN_PCA_OBJETS    = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "pca_objets.pkl"
CHEMIN_LINEAR_OOS    = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "individual_predictions_linear_oos.csv"
CHEMIN_NONLINEAR_OOS = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "individual_predictions_nonlinear_oos.csv"
CHEMIN_LOG_RETURNS   = Path(__file__).resolve().parents[2] / "data" / "monthly_log_returns.csv"
CHEMIN_SORTIE        = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "previsions_sc_svr.csv"

# ==============================================================================
# HYPERPARAMETRES DE L'ALGORITHME SC
# ==============================================================================
# Papier Section 3.2.2 : algorithme Sine-Cosine de Mirjalili (2016).
# Le papier ne precise pas la taille de population ni le nombre d'iterations
# pour SC-SVR specifiquement. Decision retenue : population=30, iterations=100,
# coherent avec les valeurs de Sermpinis 2017 (population=50-90, iter=750-1000)
# tout en limitant le temps de calcul.
SC_N_POPULATION  = 30
SC_N_ITERATIONS  = 100
SC_C_CONSTANT    = 2.0   # constante c'' dans r1 = c'' - t*(c''/T''), Mirjalili (2016)

# bornes de recherche pour C et nu du vSVR
# le papier ne les precise pas -> memes bornes que la grille SVR
SC_BORNE_C_MIN  = 0.001
SC_BORNE_C_MAX  = 1000.0
SC_BORNE_NU_MIN = 0.05
SC_BORNE_NU_MAX = 0.95

# graine aleatoire pour la reproductibilite
SEED = 42


# ==============================================================================
# SECTION 1 : CHARGEMENT DES DONNEES
# ==============================================================================
# Memes fonctions que svr.py : memes inputs, meme projection OOS.

def charger_composantes_pca(verbose: bool = True) -> dict:
    """Charge les composantes PCA in-sample (TRAIN+TEST) par facteur."""
    df_global = pd.read_csv(
        filepath_or_buffer=CHEMIN_PCA,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d", header=[0, 1]
    )
    composantes_par_facteur = {}
    for facteur in FACTEURS:
        composantes_par_facteur[facteur] = df_global[facteur].dropna(how="all")

    if verbose:
        exemple = composantes_par_facteur[FACTEURS[0]]
        print(f"Composantes PCA in-sample chargees : {exemple.shape[0]} dates, {exemple.shape[1]} composantes pour {FACTEURS[0]}")

    return composantes_par_facteur


def charger_objets_pca(verbose: bool = True) -> dict:
    """Charge les objets PCA (scaler, pca, cols_valides) sauvegardes par pca_selection.py."""
    with open(CHEMIN_PCA_OBJETS, "rb") as f:
        objets_pca = pickle.load(f)

    if verbose:
        print(f"Objets PCA charges pour {list(objets_pca.keys())}")

    return objets_pca


def projeter_previsions_oos(facteur: str, objets_pca: dict, verbose: bool = True) -> pd.DataFrame:
    """Charge les previsions individuelles OOS et les projette sur les axes PCA in-sample."""
    df_linear_oos = pd.read_csv(
        filepath_or_buffer=CHEMIN_LINEAR_OOS,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d", header=[0, 1]
    )
    df_nonlinear_oos = pd.read_csv(
        filepath_or_buffer=CHEMIN_NONLINEAR_OOS,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d", header=[0, 1]
    )

    df_oos = pd.concat([df_linear_oos[facteur], df_nonlinear_oos[facteur]], axis=1)

    if len(df_oos) == 0:
        if verbose:
            print(f"  Aucune prevision OOS disponible pour {facteur}")
        return pd.DataFrame()

    scaler, pca, cols_valides = objets_pca[facteur]
    df_oos_filtre = df_oos.reindex(columns=cols_valides, fill_value=0.0)

    for col in df_oos_filtre.columns:
        df_oos_filtre[col] = df_oos_filtre[col].fillna(value=df_oos_filtre[col].mean())

    X_oos_sc  = scaler.transform(df_oos_filtre.values)
    X_oos_pca = pca.transform(X_oos_sc)

    df_composantes_oos = pd.DataFrame(
        data=X_oos_pca,
        index=df_oos_filtre.index,
        columns=[f"PC{i+1}" for i in range(X_oos_pca.shape[1])]
    )

    if verbose:
        print(f"  Projection OOS : {df_composantes_oos.shape[0]} dates, {df_composantes_oos.shape[1]} composantes")

    return df_composantes_oos


def charger_log_rendements(verbose: bool = True) -> pd.DataFrame:
    """Charge les log-rendements mensuels."""
    df = pd.read_csv(
        filepath_or_buffer=CHEMIN_LOG_RETURNS,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d"
    )
    df = df[FACTEURS]

    if verbose:
        print(f"Log-rendements charges : {df.shape[0]} observations")

    return df


# ==============================================================================
# SECTION 2 : ALGORITHME SINE-COSINE (SC)
# ==============================================================================
# Papier Section 3.2.2, Mirjalili (2016) :
# "The modelling procedure starts with a set of random solutions and proceeds
#  to the global optima."
# Equations de mise a jour de position :
#   P_j^(t+1) = P_j^t + r1*sin(r2)*|r3*P_dest - P_j^t|  si r4 < 0.5
#   P_j^(t+1) = P_j^t + r1*cos(r2)*|r3*P_dest - P_j^t|  si r4 >= 0.5
# Ou :
#   r1 = c'' - t*(c''/T'')  : facteur d'equilibre exploration/exploitation
#   r2 ~ U[0, 2*pi]         : direction dans le cycle sinus/cosinus
#   r3 ~ U[0, 2]            : poids de la destination
#   r4 ~ U[0, 1]            : choix sine ou cosine
#   P_dest                  : meilleure position de la population courante
# La fonction fitness a maximiser est : Fitness = 1/(1 + RMSE)

def _evaluer_fitness(params: np.ndarray, X_sc: np.ndarray, y_sc: np.ndarray, scaler_y: StandardScaler, y_test: np.ndarray) -> float:
    """Evalue la fitness d'une position SC (C, nu) sur TEST. Retourne 1/(1+RMSE)."""
    c_val  = float(np.clip(params[0], SC_BORNE_C_MIN, SC_BORNE_C_MAX))
    nu_val = float(np.clip(params[1], SC_BORNE_NU_MIN, SC_BORNE_NU_MAX))

    try:
        modele = NuSVR(C=c_val, nu=nu_val, kernel="rbf")
        modele.fit(X=X_sc, y=y_sc)
        y_hat = scaler_y.inverse_transform(modele.predict(X_sc).reshape(-1, 1)).ravel()
        rmse  = float(np.sqrt(np.mean((y_hat - y_test) ** 2)))
        return 1.0 / (1.0 + rmse)
    except Exception:
        return 0.0


def optimiser_sc(X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True) -> tuple:
    """Calibre (C, nu) du vSVR par l'algorithme Sine-Cosine sur TEST. Retourne (C_opt, nu_opt, rmse_opt)."""
    # standardisation des inputs et de la cible
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_sc = scaler_x.fit_transform(X_test)
    y_sc = scaler_y.fit_transform(y_test.reshape(-1, 1)).ravel()

    np.random.seed(SEED)

    # initialisation aleatoire de la population dans l'espace [C_min, C_max] x [nu_min, nu_max]
    # positions : tableau (n_population, 2) avec colonnes [C, nu]
    positions = np.column_stack([
        np.random.uniform(SC_BORNE_C_MIN,  SC_BORNE_C_MAX,  SC_N_POPULATION),
        np.random.uniform(SC_BORNE_NU_MIN, SC_BORNE_NU_MAX, SC_N_POPULATION)
    ])

    meilleure_position = positions[0].copy()
    meilleure_fitness  = 0.0

    # evaluation initiale de la population
    for i in range(SC_N_POPULATION):
        fitness = _evaluer_fitness(
            params=positions[i], X_sc=X_sc, y_sc=y_sc,
            scaler_y=scaler_y, y_test=y_test
        )
        if fitness > meilleure_fitness:
            meilleure_fitness  = fitness
            meilleure_position = positions[i].copy()

    # boucle principale SC
    for t in range(SC_N_ITERATIONS):
        # r1 : facteur d'equilibre qui decroit lineairement -> exploration au debut, exploitation a la fin
        r1 = SC_C_CONSTANT - t * (SC_C_CONSTANT / SC_N_ITERATIONS)

        for i in range(SC_N_POPULATION):
            # variables aleatoires independantes par dimension et par individu
            r2 = np.random.uniform(0, 2 * np.pi, size=2)
            r3 = np.random.uniform(0, 2,          size=2)
            r4 = np.random.uniform(0, 1,          size=2)

            # mise a jour de position par dimension (sine ou cosine selon r4)
            nouvelle_position = positions[i].copy()
            for d in range(2):
                distance = abs(r3[d] * meilleure_position[d] - positions[i][d])
                if r4[d] < 0.5:
                    nouvelle_position[d] = positions[i][d] + r1 * np.sin(r2[d]) * distance
                else:
                    nouvelle_position[d] = positions[i][d] + r1 * np.cos(r2[d]) * distance

            # clamping dans les bornes
            nouvelle_position[0] = np.clip(nouvelle_position[0], SC_BORNE_C_MIN,  SC_BORNE_C_MAX)
            nouvelle_position[1] = np.clip(nouvelle_position[1], SC_BORNE_NU_MIN, SC_BORNE_NU_MAX)

            positions[i] = nouvelle_position

            # evaluation et mise a jour de la meilleure solution
            fitness = _evaluer_fitness(
                params=positions[i], X_sc=X_sc, y_sc=y_sc,
                scaler_y=scaler_y, y_test=y_test
            )
            if fitness > meilleure_fitness:
                meilleure_fitness  = fitness
                meilleure_position = positions[i].copy()

    c_opt  = float(np.clip(meilleure_position[0], SC_BORNE_C_MIN,  SC_BORNE_C_MAX))
    nu_opt = float(np.clip(meilleure_position[1], SC_BORNE_NU_MIN, SC_BORNE_NU_MAX))
    rmse_opt = 1.0 / meilleure_fitness - 1.0

    if verbose:
        print(f"  SC : C={c_opt:.4f}, nu={nu_opt:.4f}, RMSE_test={rmse_opt:.6f}")

    return c_opt, nu_opt, rmse_opt


# ==============================================================================
# SECTION 3 : PREVISIONS SC-SVR SUR OOS
# ==============================================================================
# Une fois (C, nu) trouves par SC, le vSVR est entraine sur TEST et applique
# sur OOS exactement comme dans svr.py. La seule difference avec SVR est la
# methode de calibration des parametres.

def prevoir_scsvr_oos(X_test: np.ndarray, y_test: np.ndarray, X_oos: np.ndarray, c: float, nu: float) -> np.ndarray:
    """Entraine le vSVR sur TEST avec (C, nu) optimises par SC et predit sur OOS."""
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

def prevoir_scsvr_facteur(facteur: str, composantes_insample: pd.DataFrame, composantes_oos: pd.DataFrame, df_log: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """Calibre le vSVR par SC sur TEST et produit les previsions OOS pour un facteur."""
    if verbose:
        print(f"\n  Facteur : {facteur}")

    # X_test : composantes PCA sur TEST uniquement
    masque_test = (composantes_insample.index >= TEST_START) & (composantes_insample.index <= TEST_END)
    X_test  = composantes_insample.loc[masque_test].values
    X_oos   = composantes_oos.values
    idx_oos = composantes_oos.index

    # vraies valeurs TEST
    masque_test_log = (df_log.index >= TEST_START) & (df_log.index <= TEST_END)
    y_test = df_log.loc[masque_test_log, facteur].values

    if len(X_test) == 0 or len(X_oos) == 0:
        if verbose:
            print(f"  Pas de donnees TEST ou OOS pour {facteur}")
        return pd.Series(dtype=float, name=f"SCSVR_{facteur}")

    # calibration par SC sur TEST
    c_opt, nu_opt, rmse_test = optimiser_sc(X_test=X_test, y_test=y_test, verbose=verbose)

    # previsions OOS
    previsions = prevoir_scsvr_oos(
        X_test=X_test, y_test=y_test,
        X_oos=X_oos, c=c_opt, nu=nu_opt
    )

    serie_previsions = pd.Series(data=previsions, index=idx_oos, name=f"SCSVR_{facteur}")

    if verbose:
        print(f"  SC-SVR {facteur} : {len(serie_previsions)} previsions OOS")

    return serie_previsions


# ==============================================================================
# SECTION 5 : PIPELINE COMPLET POUR LES 5 FACTEURS
# ==============================================================================

def executer_previsions_scsvr(verbose: bool = True) -> pd.DataFrame:
    """Calibre et execute le SC-SVR pour les 5 facteurs. Sauvegarde et retourne les previsions OOS."""
    print("ETAPE 02_FORECASTING SC-SVR ==============================") if verbose else None

    composantes_par_facteur = charger_composantes_pca(verbose=verbose)
    objets_pca              = charger_objets_pca(verbose=verbose)
    df_log                  = charger_log_rendements(verbose=verbose)

    previsions_par_facteur = {}

    for facteur in FACTEURS:
        composantes_oos = projeter_previsions_oos(
            facteur=facteur, objets_pca=objets_pca, verbose=verbose
        )
        serie_prev = prevoir_scsvr_facteur(
            facteur=facteur,
            composantes_insample=composantes_par_facteur[facteur],
            composantes_oos=composantes_oos,
            df_log=df_log,
            verbose=verbose
        )
        previsions_par_facteur[facteur] = serie_prev

    df_previsions = pd.DataFrame(data=previsions_par_facteur)
    df_previsions.index.name = "date"

    CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    df_previsions.to_csv(path_or_buf=CHEMIN_SORTIE, date_format="%Y-%m-%d")

    if verbose:
        print(f"\nPrevisions SC-SVR sauvegardees : {CHEMIN_SORTIE}")
        print(f"Dimensions : {df_previsions.shape[0]} dates OOS x {df_previsions.shape[1]} facteurs")
        print("ETAPE 02_FORECASTING SC-SVR END ==========================")

    return df_previsions


# ==============================================================================
# SECTION 6 : CHARGEMENT
# ==============================================================================

def charger_previsions_scsvr(verbose: bool = True) -> pd.DataFrame:
    """Charge les previsions SC-SVR sauvegardees depuis le CSV."""
    df = pd.read_csv(
        filepath_or_buffer=CHEMIN_SORTIE,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d"
    )

    if verbose:
        print(f"Previsions SC-SVR chargees : {df.shape[0]} dates, {df.shape[1]} facteurs")

    return df 


if __name__ == "__main__":
    df_previsions = executer_previsions_scsvr(verbose=True)
    a = True