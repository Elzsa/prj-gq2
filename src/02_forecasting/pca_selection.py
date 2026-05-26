# src/02_forecasting/pca_selection.py

import sys
from pathlib import Path

# ajout de la racine du projet au sys.path pour permettre les imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.splits import TRAIN_START, TRAIN_END, TEST_START, TEST_END

# noms des cinq facteurs Fama-French
FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]

# seuil de variance expliquee cumulee retenu par le papier (Section 3.1)
SEUIL_VARIANCE = 0.95

# chemins des previsions individuelles produites par linear.py et nonlinear.py
CHEMIN_LINEAR    = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "individual_predictions_linear.csv"
CHEMIN_NONLINEAR = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "individual_predictions_nonlinear.csv"

# chemin de sortie des composantes PCA (inputs de SVR, SC-SVR, DMA)
CHEMIN_SORTIE_PCA       = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "pca_components.csv"

# chemin de sortie du meilleur modele individuel par facteur (benchmark)
CHEMIN_SORTIE_MEILLEUR  = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "best_individual.csv"

# chemin de sortie du rapport PCA (nb composantes, variance expliquee, meilleur modele)
CHEMIN_SORTIE_RAPPORT   = Path(__file__).resolve().parents[2] / "results" / "tables" / "pca_rapport.csv"

# chemin de sortie de la Table 3 (reproduction du papier, Section 3.1)
CHEMIN_TABLE_3 = Path(__file__).resolve().parents[2] / "results" / "tables" / "table_3.csv"

# metriques d'evaluation pour selectionner le meilleur modele individuel
# le papier utilise MAE, RMSE, MAPE, Theil-U (Table 4)
# decision retenue : RMSE comme critere de selection du meilleur modele
# (coherent avec la fitness SC-SVR = 1/(1+RMSE) et la metrique DM du papier)


# ==============================================================================
# SECTION 1 : CHARGEMENT DES PREVISIONS INDIVIDUELLES
# ==============================================================================

def charger_previsions_individuelles(verbose: bool = True) -> dict:
    """Charge et concatene les previsions lineaires et non lineaires par facteur."""
    # chargement des previsions lineaires (multi-index colonnes : facteur, modele)
    df_linear = pd.read_csv(
        filepath_or_buffer=CHEMIN_LINEAR,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
        header=[0, 1]
    )

    # chargement des previsions non lineaires (multi-index colonnes : facteur, modele)
    df_nonlinear = pd.read_csv(
        filepath_or_buffer=CHEMIN_NONLINEAR,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
        header=[0, 1]
    )

    previsions_par_facteur = {}

    for facteur in FACTEURS:
        df_lin    = df_linear[facteur]
        df_nonlin = df_nonlinear[facteur]

        # concatenation horizontale : 290 colonnes lineaires + 9 colonnes non lineaires = 299
        # le papier annonce 328, l'ecart vient des variantes non implementees (nonlinear.py)
        df_complet = pd.concat(objs=[df_lin, df_nonlin], axis=1)

        # periode in-sample complete TRAIN+TEST (1965-1999) pour la PCA
        # papier Section 3.1 : "individual forecasts in-sample"
        masque = (df_complet.index >= TRAIN_START) & (df_complet.index <= TEST_END)
        df_complet = df_complet.loc[masque]

        previsions_par_facteur[facteur] = df_complet

    if verbose:
        exemple = previsions_par_facteur[FACTEURS[0]]
        print(f"Previsions individuelles chargees : {exemple.shape[0]} observations, {exemple.shape[1]} modeles par facteur")
        print(f"Periode : {exemple.index.min().date()} - {exemple.index.max().date()}")

    return previsions_par_facteur


# ==============================================================================
# SECTION 2 : CALCUL DU MEILLEUR MODELE INDIVIDUEL PAR FACTEUR
# ==============================================================================
# Papier Section 3.1 : "the RW and the best predictor of each case (bold)
# play the role of our benchmarks."
# Le meilleur modele individuel est identifie en gras dans la Table 3.
# Il sert de benchmark pour evaluer les methodes de combinaison (SVR, SC-SVR, DMA).
# Critere de selection retenu : RMSE minimale sur la periode TEST.

def identifier_meilleur_modele(df_previsions: pd.DataFrame, serie_reelle: pd.Series, verbose: bool = True) -> tuple:
    """Identifie le modele individuel avec la RMSE minimale sur TEST. Retourne (nom_modele, serie_previsions)."""
    # filtrage sur les dates communes entre previsions et serie reelle
    dates_communes = df_previsions.index.intersection(serie_reelle.index)
    df_prev        = df_previsions.loc[dates_communes]
    y_reel         = serie_reelle.loc[dates_communes].values

    rmse_par_modele = {}
    for col in df_prev.columns:
        serie_col = df_prev[col].dropna()
        # ne considerer que les modeles avec suffisamment de previsions valides
        if len(serie_col) < len(dates_communes) * 0.9:
            continue
        y_hat = df_prev.loc[serie_col.index, col].values
        y_ref  = serie_reelle.loc[serie_col.index].values
        rmse   = float(np.sqrt(np.mean((y_hat - y_ref) ** 2)))
        rmse_par_modele[col] = rmse

    if len(rmse_par_modele) == 0:
        raise ValueError("Aucun modele individuel valide trouve pour ce facteur.")

    nom_meilleur   = min(rmse_par_modele, key=rmse_par_modele.get)
    rmse_meilleur  = rmse_par_modele[nom_meilleur]
    serie_meilleur = df_previsions[nom_meilleur]

    if verbose:
        print(f"  Meilleur modele individuel : {nom_meilleur} (RMSE={rmse_meilleur:.6f})")

    return nom_meilleur, serie_meilleur, rmse_meilleur


# ==============================================================================
# SECTION 3 : APPLICATION DE L'ACP PAR FACTEUR
# ==============================================================================
# Papier Section 3.1 :
# "the principal component analysis (PCA) is used in order to discard highly
#  correlated variables, while accounting for the 95% of the total variance."
#
# Procedure retenue :
# 1. Supprimer les colonnes avec trop de NaN (moins de 90% de valeurs valides)
# 2. Imputer les NaN restants par la moyenne de la colonne
# 3. Standardiser (zero moyenne, unit variance) : requis par la PCA
# 4. Appliquer PCA avec n_components=0.95 (seuil du papier)
# 5. Les composantes retenues sont les inputs de SVR, SC-SVR, DMA
#
# Le papier dit "individual forecasts in-sample" (Section 3.1).
# Decision retenue : periode TRAIN+TEST (1965-1999), soit 420 observations.
# La PCA sur la periode complete garantit une structure de correlation plus stable
# et permet aux modeles non lineaires (estimes sur TRAIN) d'avoir des fittedvalues
# sur toute la periode in-sample.

def appliquer_pca_facteur(df_previsions: pd.DataFrame, nom_facteur: str, verbose: bool = True) -> tuple:
    """Applique la PCA sur les previsions d'un facteur et retourne (composantes, objet PCA, scaler, colonnes_retenues)."""
    n_obs     = df_previsions.shape[0]
    n_modeles = df_previsions.shape[1]

    # etape 1 : supprimer les colonnes avec trop de NaN
    seuil_nan     = 0.9  # au moins 90% de valeurs valides requises
    cols_valides  = [col for col in df_previsions.columns
                     if df_previsions[col].notna().sum() >= seuil_nan * n_obs]
    df_filtre     = df_previsions[cols_valides].copy()
    n_supprimes   = n_modeles - len(cols_valides)

    if verbose:
        print(f"  Colonnes supprimees (trop de NaN) : {n_supprimes} / {n_modeles}")
        print(f"  Colonnes retenues avant PCA : {len(cols_valides)}")

    # etape 2 : imputer les NaN restants par la moyenne de chaque colonne
    for col in df_filtre.columns:
        moyenne = df_filtre[col].mean()
        df_filtre[col] = df_filtre[col].fillna(value=moyenne)

    # etape 3 : standardisation
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(df_filtre.values)

    # etape 4 : PCA avec seuil de variance
    # svd_solver='full' : plus stable sur des matrices rectangulaires larges
    pca   = PCA(n_components=SEUIL_VARIANCE, svd_solver="full")
    X_pca = pca.fit_transform(X_sc)

    n_composantes       = pca.n_components_
    variance_expliquee  = float(pca.explained_variance_ratio_.cumsum()[-1])

    if verbose:
        print(f"  Composantes PCA retenues (95% variance) : {n_composantes}")
        print(f"  Variance expliquee cumulee : {variance_expliquee:.4f}")

    # construction du DataFrame de sortie : index = dates TEST, colonnes = PC1, PC2, ...
    noms_composantes = [f"PC{i+1}" for i in range(n_composantes)]
    df_composantes   = pd.DataFrame(data=X_pca, index=df_filtre.index, columns=noms_composantes)

    return df_composantes, pca, scaler, cols_valides, n_composantes, variance_expliquee


# ==============================================================================
# SECTION 4 : PIPELINE COMPLET POUR LES 5 FACTEURS
# ==============================================================================

def executer_pca_selection(df_log: pd.DataFrame, verbose: bool = True) -> dict:
    """Applique la PCA sur les previsions de chaque facteur et sauvegarde les composantes et le meilleur modele."""
    print("ETAPE 02_FORECASTING PCA SELECTION =======================") if verbose else None

    previsions_par_facteur = charger_previsions_individuelles(verbose=verbose)

    # dictionnaires de sortie
    composantes_par_facteur    = {}   # facteur -> DataFrame(PC1, PC2, ...) sur TEST
    meilleur_par_facteur       = {}   # facteur -> Series (meilleur modele individuel)
    pca_objects                = {}   # facteur -> objet PCA ajuste (pour generer_table_3)
    cols_valides_par_facteur   = {}   # facteur -> liste des colonnes retenues avant PCA
    rapport_lignes             = []   # pour le CSV de rapport

    for facteur in FACTEURS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Facteur : {facteur}")
            print(f"{'='*60}")

        df_prev        = previsions_par_facteur[facteur]
        serie_reelle   = df_log[facteur].loc[TEST_START:TEST_END]

        # identification du meilleur modele individuel (benchmark)
        nom_meilleur, serie_meilleur, rmse_meilleur = identifier_meilleur_modele(
            df_previsions=df_prev,
            serie_reelle=serie_reelle,
            verbose=verbose
        )
        meilleur_par_facteur[facteur] = serie_meilleur.rename(nom_meilleur)

        # application de la PCA
        df_composantes, pca, scaler, cols_valides, n_comp, var_exp = appliquer_pca_facteur(
            df_previsions=df_prev,
            nom_facteur=facteur,
            verbose=verbose
        )
        composantes_par_facteur[facteur]  = df_composantes
        pca_objects[facteur]              = pca
        cols_valides_par_facteur[facteur] = cols_valides

        rapport_lignes.append({
            "facteur":              facteur,
            "n_modeles_total":      df_prev.shape[1],
            "n_modeles_valides":    len(cols_valides),
            "n_composantes_pca":    n_comp,
            "variance_expliquee":   round(var_exp, 4),
            "meilleur_modele":      nom_meilleur,
            "rmse_meilleur":        round(rmse_meilleur, 6),
        })

    # --- sauvegarde des composantes PCA (multi-index : facteur, composante) ---
    df_composantes_global = pd.concat(objs=composantes_par_facteur, axis=1)
    df_composantes_global.columns.names = ["facteur", "composante"]
    CHEMIN_SORTIE_PCA.parent.mkdir(parents=True, exist_ok=True)
    df_composantes_global.to_csv(path_or_buf=CHEMIN_SORTIE_PCA, date_format="%Y-%m-%d")

    # --- sauvegarde du meilleur modele individuel par facteur ---
    df_meilleur = pd.DataFrame(data=meilleur_par_facteur)
    df_meilleur.index.name = "date"
    CHEMIN_SORTIE_MEILLEUR.parent.mkdir(parents=True, exist_ok=True)
    df_meilleur.to_csv(path_or_buf=CHEMIN_SORTIE_MEILLEUR, date_format="%Y-%m-%d")

    # --- sauvegarde du rapport ---
    df_rapport = pd.DataFrame(data=rapport_lignes).set_index("facteur")
    CHEMIN_SORTIE_RAPPORT.parent.mkdir(parents=True, exist_ok=True)
    df_rapport.to_csv(path_or_buf=CHEMIN_SORTIE_RAPPORT)

    # generation de la Table 3
    generer_table_3(
        pca_objects=pca_objects,
        cols_valides_par_facteur=cols_valides_par_facteur,
        meilleur_par_facteur=meilleur_par_facteur,
        verbose=verbose
    )

    if verbose:
        print(f"\n{'='*60}")
        print("RAPPORT PCA :")
        print(df_rapport.to_string())
        print(f"\nComposantes PCA sauvegardees : {CHEMIN_SORTIE_PCA}")
        print(f"Meilleur modele par facteur sauvegarde : {CHEMIN_SORTIE_MEILLEUR}")
        print(f"Rapport PCA sauvegarde : {CHEMIN_SORTIE_RAPPORT}")
        print("ETAPE 02_FORECASTING PCA SELECTION END ===================")

    return composantes_par_facteur, meilleur_par_facteur, pca_objects, cols_valides_par_facteur


# ==============================================================================
# SECTION 5 : FONCTIONS DE CHARGEMENT
# ==============================================================================

def charger_composantes_pca(verbose: bool = True) -> dict:
    """Charge les composantes PCA sauvegardees depuis le CSV multi-index."""
    df_global = pd.read_csv(
        filepath_or_buffer=CHEMIN_SORTIE_PCA,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
        header=[0, 1]
    )

    composantes_par_facteur = {}
    for facteur in FACTEURS:
        composantes_par_facteur[facteur] = df_global[facteur]

    if verbose:
        exemple = composantes_par_facteur[FACTEURS[0]]
        print(f"Composantes PCA chargees : {exemple.shape[0]} dates, {exemple.shape[1]} composantes par facteur")

    return composantes_par_facteur


def charger_meilleur_individuel(verbose: bool = True) -> pd.DataFrame:
    """Charge les previsions du meilleur modele individuel par facteur."""
    df = pd.read_csv(
        filepath_or_buffer=CHEMIN_SORTIE_MEILLEUR,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d"
    )

    if verbose:
        print(f"Meilleur modele individuel charge : {df.shape[0]} dates, facteurs={list(df.columns)}")

    return df


# ==============================================================================
# SECTION 6 : GENERATION DE LA TABLE 3 DU PAPIER
# ==============================================================================
# Papier Table 3 (page 9) : "Best predictors' set"
# Colonnes : un facteur par colonne (MKT, SMB, HML, RMW, CMA)
# Lignes : noms des modeles retenus apres PCA, meilleur en gras (non disponible en CSV)
#
# Methode pour retrouver les noms des modeles a partir de la PCA :
# Pour chaque composante retenue, on identifie le modele avec le chargement absolu
# le plus eleve (valeur absolue des loadings). Les modeles les plus representatifs
# de l'ensemble des composantes constituent le "best predictors set" de la Table 3.
#
# Ambiguite du papier : il n'est pas precise comment les noms sont selectionnes
# a partir des composantes PCA. Decision retenue : pour chaque composante PC_k,
# on retient le modele avec le loading |w_ik| maximal. On deduplique ensuite.

def generer_table_3(pca_objects: dict, cols_valides_par_facteur: dict, meilleur_par_facteur: dict, verbose: bool = True) -> pd.DataFrame:
    """Genere et sauvegarde la Table 3 du papier : best predictors set par facteur.

    Methode : pour chaque composante PCA retenue, on identifie le modele original
    avec le loading absolu maximal sur cette composante (|w_ik| maximal).
    Ce modele est le representant de la composante dans la Table 3.
    On deduplique ensuite pour eviter qu'un meme modele apparaisse deux fois.

    Cette logique est coherente avec la PCA : deux modeles tres correles (ex: SMA(24)
    et SMA(25)) seront captures par la meme composante et donneront le meme representant,
    garantissant la diversite observee dans la Table 3 du papier.

    Le meilleur modele individuel (RMSE minimale sur TEST) est marque [BEST].
    """
    donnees_table = {}

    for facteur in FACTEURS:
        pca        = pca_objects[facteur]
        cols       = cols_valides_par_facteur[facteur]
        nom_best   = meilleur_par_facteur[facteur].name

        # loadings : matrice (n_composantes, n_modeles_valides)
        # pca.components_[k, i] = contribution du modele i a la composante k
        loadings = pca.components_

        noms_retenus = []
        for k in range(loadings.shape[0]):
            # modele avec le loading absolu maximal sur la composante k
            idx_max    = int(np.argmax(np.abs(loadings[k])))
            nom_modele = cols[idx_max]
            # deduplification : un modele ne peut representer qu'une seule composante
            if nom_modele not in noms_retenus:
                noms_retenus.append(nom_modele)

        # si le meilleur modele n'est pas dans la liste PCA, on l'ajoute en premier
        # cela garantit que [BEST] apparait toujours dans la Table 3
        if nom_best not in noms_retenus:
            noms_retenus.insert(0, nom_best)
            if verbose:
                print(f"  {facteur} : meilleur modele {nom_best} force en tete (absent des loadings PCA)")

        # marquer le meilleur modele avec [BEST] (gras dans le papier)
        noms_affiches = [f"{n} [BEST]" if n == nom_best else n for n in noms_retenus]
        donnees_table[facteur] = noms_affiches

        if verbose:
            print(f"  {facteur} : {len(noms_retenus)} modeles -> {noms_retenus}")

    # padding pour DataFrame rectangulaire (longueurs differentes par facteur)
    longueur_max = max(len(v) for v in donnees_table.values())
    for facteur in FACTEURS:
        while len(donnees_table[facteur]) < longueur_max:
            donnees_table[facteur].append("")

    df_table_3 = pd.DataFrame(data=donnees_table)
    df_table_3.index = [f"Modele_{i+1}" for i in range(longueur_max)]
    df_table_3.index.name = "Rang"

    CHEMIN_TABLE_3.parent.mkdir(parents=True, exist_ok=True)
    df_table_3.to_csv(path_or_buf=CHEMIN_TABLE_3)

    if verbose:
        print(f"\nTable 3 sauvegardee : {CHEMIN_TABLE_3}")
        print(df_table_3.to_string())

    return df_table_3


if __name__ == "__main__":
    import importlib.util as _ilu

    # src/01_data/preprocess.py : le nom du dossier "01_data" contient un chiffre,
    # ce qui rend l'import Python standard invalide -> on utilise importlib
    _chemin_preprocess = Path(__file__).resolve().parents[2] / "src" / "01_data" / "preprocess.py"
    _spec = _ilu.spec_from_file_location(name="preprocess", location=_chemin_preprocess)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    df_log = _mod.executer_preprocessing(verbose=False)
    composantes, meilleur, pca_objects, cols_valides = executer_pca_selection(df_log=df_log, verbose=True)
    a = True