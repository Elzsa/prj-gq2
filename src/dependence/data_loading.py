# src/dependency_structure/data_loading.py

from pathlib import Path

import pandas as pd

from config import config_dependance


def load_data(path: str | Path = config_dependance.DATA_PATH):
    """
    Charge et met en forme les données depuis le fichier Excel Fama-French.

    Paramètres
    ----------
    path : str | Path
        Chemin vers le fichier Excel. Par défaut : config.DATA_PATH.

    Retourne
    --------
    factors : pd.DataFrame
        Série complète des 5 facteurs (toutes les dates disponibles).
    in_sample : pd.DataFrame
        Sous-période in-sample  [config.IN_SAMPLE_START → config.IN_SAMPLE_END].
    out_sample : pd.DataFrame
        Sous-période out-of-sample [config.OUT_SAMPLE_START → config.OUT_SAMPLE_END].
    """
    path = Path(path)

    # Lire le fichier Excel : les vrais headers sont après 4 lignes
    df = pd.read_excel(path, skiprows=4)

    # Garder uniquement les 7 colonnes utiles
    df = df.iloc[:, :7]

    # Renommer les colonnes
    df.columns = ["Date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]

    # Convertir la colonne Date en numérique
    df["Date"] = pd.to_numeric(df["Date"], errors="coerce")

    # Garder uniquement les dates mensuelles au format YYYYMM
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].astype(int).astype(str)
    df = df[df["Date"].str.fullmatch(r"\d{6}")]

    # Convertir la date en datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")

    # Convertir les facteurs en float
    df[config_dependance.FACTORS] = df[config_dependance.FACTORS].apply(
        pd.to_numeric, errors="coerce"
    )

    # Garder uniquement les 5 facteurs et supprimer les NaN
    factors = df[config_dependance.FACTORS].dropna()

    # Les données sont en % → diviser par 100
    factors = factors / 100

    # Séparation in-sample / out-of-sample selon config
    in_sample = factors.loc[
        config_dependance.IN_SAMPLE_START : config_dependance.IN_SAMPLE_END
    ]
    out_sample = factors.loc[
        config_dependance.OUT_SAMPLE_START : config_dependance.OUT_SAMPLE_END
    ]

    return factors, in_sample, out_sample


def _load_momentum(
    path: str | Path = config_dependance.MOMENTUM_DATA_PATH,
) -> pd.DataFrame:
    """
    Charge le facteur Momentum (MOM) depuis le fichier Fama-French.
    Retourne un DataFrame mensuel avec colonne 'MOM', en décimal.
    """
    df = pd.read_excel(
        Path(path), skiprows=14, header=None, index_col=False, usecols=[0, 1]
    )
    df.columns = ["Date", "MOM"]

    df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].astype(int).astype(str)
    df = df[df["Date"].str.fullmatch(r"\d{6}")]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")

    df["MOM"] = pd.to_numeric(df["MOM"].astype(str).str.strip(), errors="coerce") / 100

    return df.dropna()


def load_data_ext(
    ff5_path: str | Path = config_dependance.DATA_PATH,
    mom_path: str | Path = config_dependance.MOMENTUM_DATA_PATH,
):
    """
    Charge les 6 facteurs (FF5 + UMD) pour l'extension Momentum.

    Retourne
    --------
    factors_ext : pd.DataFrame  — série complète des 6 facteurs
    in_sample   : pd.DataFrame  — [IN_SAMPLE_START → IN_SAMPLE_END]
    out_sample  : pd.DataFrame  — [OUT_SAMPLE_START → OUT_SAMPLE_END]
    """
    factors_ff5, _, _ = load_data(ff5_path)
    mom = _load_momentum(mom_path)

    # Inner join : intersection des dates (FF5 commence en 1965-01)
    factors_ext = factors_ff5.join(mom, how="inner")[
        config_dependance.FACTORS_EXT
    ].dropna()

    in_sample = factors_ext.loc[
        config_dependance.IN_SAMPLE_START : config_dependance.IN_SAMPLE_END
    ]
    out_sample = factors_ext.loc[
        config_dependance.OUT_SAMPLE_START : config_dependance.OUT_SAMPLE_END
    ]
    return factors_ext, in_sample, out_sample
