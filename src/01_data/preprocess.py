# src/01_data/preprocess.py

import sys
from pathlib import Path

# ajout de la racine du projet au sys.path pour permettre les imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from config.splits import TOTAL_START, TOTAL_END, valider_splits

# chemins des fichiers # TODO : déplacer dans le fichier des variables globales ? 
CHEMIN_RAW    = Path(__file__).resolve().parents[2] / "data" / "F-F_Research_Data_5_Factors_2x3.csv"
CHEMIN_SORTIE = Path(__file__).resolve().parents[2] / "data" / "monthly_log_returns.csv"

# renommage des colonnes brutes vers les noms du papier
RENOMMAGE_COLONNES = {
    "Mkt-RF": "MKT",
    "SMB":    "SMB",
    "HML":    "HML",
    "RMW":    "RMW",
    "CMA":    "CMA",
}


def charger_donnees_brutes(chemin: Path, verbose: bool = True) -> pd.DataFrame:
    """Charge le fichier CSV journalier brut de Ken French et retourne un DataFrame avec index datetime journalier."""
    # lecture en sautant les 4 lignes d'en-tête descriptif de Ken French
    df = pd.read_csv(
        filepath_or_buffer=chemin,
        skiprows=4,
        index_col=0,
        dtype=str        # toutes les colonnes lues en string pour éviter toute conversion automatique
    )

    n_avant = len(df)

    # suppression de la colonne RF (taux sans risque, non utilisée dans le papier)
    df = df.drop(columns=["RF"], errors="ignore")

    # renommage des colonnes
    df = df.rename(columns=RENOMMAGE_COLONNES)

    # suppression des lignes dont l'index n'est pas au format YYYYMM (6 chiffres)
    df = df[df.index.str.match(r"^\d{6}$", na=False)]

    # conversion des colonnes en float
    df = df.apply(pd.to_numeric, errors="coerce")

    # suppression des lignes avec valeurs manquantes
    df = df.dropna()
    n_apres = len(df)

    # conversion de l'index YYYYMM (string) en datetime fin de mois
    df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "date"

    if verbose:
        n_drops = n_avant - n_apres
        print(f"Données brutes chargées : {n_avant} observations")
        print(f"Données après drop : {n_apres} observations, {df.shape[1]} facteurs")
        print(f"Lignes supprimées (dropna) : {n_drops}")
        print(f"Période disponible : {df.index.min().date()} - {df.index.max().date()}")
        
    return df


def filtrer_periode(df: pd.DataFrame, date_debut: str, date_fin: str, verbose: bool = True) -> pd.DataFrame:
    """Filtre le DataFrame sur la période [date_debut, date_fin] incluse."""
    masque = (df.index >= date_debut) & (df.index <= date_fin)
    df_filtre = df.loc[masque].copy()

    if verbose:
        print(f"Filtrage sur {date_debut} - {date_fin} : {df_filtre.shape[0]} observations retenues")

    return df_filtre


def convertir_en_log_rendements(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Convertit les rendements simples en pourcentage en log-rendements."""
    # les données French sont en pourcentage : division par 100 pour obtenir des rendements décimaux
    r = df / 100.0

    # log-rendement : ln(1 + r)
    log_r = np.log(1.0 + r)

    if verbose:
        print("Conversion en log-rendements effectuée.")

    return log_r


def afficher_observations_par_periode(df: pd.DataFrame, verbose: bool = True) -> None:
    """Affiche le nombre d'observations journalières par période (TRAIN, TEST, OOS)."""
    from config.splits import TRAIN_START, TRAIN_END, TEST_START, TEST_END, OOS_START, OOS_END

    periodes = {
        "TRAIN": (TRAIN_START, TRAIN_END),
        "TEST":  (TEST_START,  TEST_END),
        "OOS":   (OOS_START,   OOS_END),
    }

    if verbose:
        print("\nNombre d'observations par période :")
        for nom, (debut, fin) in periodes.items():
            masque = (df.index >= debut) & (df.index <= fin)
            n = masque.sum()
            print(f"  {nom} ({debut} - {fin}) : {n} observations")


def generer_table_2(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Génère la Table 2 du papier : nombre d'observations par période et sauvegarde en CSV."""
    from config.splits import TRAIN_START, TRAIN_END, TEST_START, TEST_END, OOS_START, OOS_END

    periodes = {
        "Total dataset":        (TOTAL_START, TOTAL_END),
        "In-sample dataset":    (TRAIN_START, TEST_END),
        "Training dataset":     (TRAIN_START, TRAIN_END),
        "Test dataset":         (TEST_START,  TEST_END),
        "Out-of-sample dataset":(OOS_START,   OOS_END),
    }

    lignes = []
    for nom, (debut, fin) in periodes.items():
        masque = (df.index >= debut) & (df.index <= fin)
        n = int(masque.sum())
        lignes.append({"Dataset": nom, "Start date": debut, "End date": fin, "Trading days": n})

    table_2 = pd.DataFrame(data=lignes).set_index("Dataset")

    chemin_sortie = Path(__file__).resolve().parents[2] / "results" / "tables" / "table_2.csv"
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    table_2.to_csv(path_or_buf=chemin_sortie)

    if verbose:
        print(f"Table 2 sauvegardée : {chemin_sortie}")

    return table_2


def sauvegarder_csv(df: pd.DataFrame, chemin: Path, verbose: bool = True) -> None:
    """Sauvegarde le DataFrame en CSV à l'emplacement indiqué."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=chemin, date_format="%Y-%m-%d")

    if verbose:
        print(f"Fichier sauvegardé : {chemin} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")


def executer_preprocessing(verbose: bool = True) -> pd.DataFrame:
    """Exécute le pipeline complet : chargement, filtrage, conversion et sauvegarde des log-rendements."""
    
    print("ETAPE 01_DATA PRE PROCESSING ============================") if verbose else None
    valider_splits(verbose=verbose)

    df_brut   = charger_donnees_brutes(chemin=CHEMIN_RAW, verbose=verbose)
    df_filtre = filtrer_periode(df=df_brut, date_debut=TOTAL_START, date_fin=TOTAL_END, verbose=verbose)
    df_log    = convertir_en_log_rendements(df=df_filtre, verbose=verbose)
    afficher_observations_par_periode(df=df_log, verbose=verbose)
    generer_table_2(df=df_log, verbose=verbose)

    sauvegarder_csv(df=df_log, chemin=CHEMIN_SORTIE, verbose=verbose)
    print("ETAPE 01_DATA PRE PROCESSING END ========================") if verbose else None
    return df_log


if __name__ == "__main__":
    df_log = executer_preprocessing(verbose=True)
    a = True