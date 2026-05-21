# src/01_data/validation.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

CHEMIN_LOG_RETURNS = Path(__file__).resolve().parents[2] / "data" / "monthly_log_returns.csv"
CHEMIN_TABLES      = Path(__file__).resolve().parents[2] / "results" / "tables"
FACTEURS           = ["MKT", "SMB", "HML", "RMW", "CMA"]
CHEMIN_RAW = Path(__file__).resolve().parents[2] / "data" / "F-F_Research_Data_5_Factors_2x3.csv"
RENOMMAGE_COLONNES = {
    "Mkt-RF": "MKT",
    "SMB":    "SMB",
    "HML":    "HML",
    "RMW":    "RMW",
    "CMA":    "CMA",
}

def charger_rendements(date_debut: str, date_fin: str, log: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Charge les rendements mensuels (bruts ou log) filtrés sur [date_debut, date_fin]."""
    from config.splits import TOTAL_START, TOTAL_END

    chemin = CHEMIN_LOG_RETURNS if log else CHEMIN_RAW

    df = pd.read_csv(
        filepath_or_buffer=chemin,
        index_col=0,
        parse_dates=True,
        skiprows=0 if log else 4,  # le fichier brut Ken French a 4 lignes d'en-tête
        date_format="%Y-%m-%d",
        dtype=None if log else str
    )

    if not log:
        # mêmes étapes de nettoyage que dans preprocess.py
        df = df.drop(columns=["RF"], errors="ignore")
        df = df.rename(columns=RENOMMAGE_COLONNES)
        df = df[df.index.str.match(r"^\d{6}$", na=False)]
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()
        df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
        df.index.name = "date"

    # filtrage sur la période demandée
    masque = (df.index >= date_debut) & (df.index <= date_fin)
    df = df.loc[masque].copy()

    if verbose:
        print(f"Log-rendements chargés période : {df.index.min().date()} - {df.index.max().date()} : {df.shape[0]} observations, {df.shape[1]} facteurs")

    return df


def formater_pvalue(pval: float) -> str:
    """Formate une p-value en ajoutant *** si rejet au seuil de 1%."""
    etoiles = "***" if pval < 0.01 else ""
    return f"{pval:.3f}{etoiles}"


def generer_table_1a(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Génère le Panel A de la Table 1 : statistiques descriptives des log-rendements en pourcentage."""
    # conversion en pourcentage pour correspondre aux unités du papier
    df_pct = df[FACTEURS] 

    lignes = {}
    for col in FACTEURS:
        serie = df_pct[col].dropna()

        # test de Jarque-Bera
        _, pval_jb = stats.jarque_bera(x=serie)

        # test ADF
        resultat_adf = adfuller(x=serie, autolag="AIC")
        pval_adf = resultat_adf[1]

        lignes[col] = {
            "Mean":               round(serie.mean(), 3),
            "Median":             round(serie.median(), 3),
            "Standard deviation": round(serie.std(), 3),
            "Skewness":           round(serie.skew(), 3),
            "Kurtosis":           round(stats.kurtosis(a=serie, fisher=False), 3),
            "Jarque-Bera (p value)": formater_pvalue(pval=pval_jb),
            "ADF (p value)":         formater_pvalue(pval=pval_adf),
        }

    table_1a = pd.DataFrame(data=lignes)
    table_1a.index.name = "Ticker"
    chemin_sortie = CHEMIN_TABLES / "table_1a.csv"
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    table_1a.to_csv(path_or_buf=chemin_sortie)

    if verbose:
        print(f"Table 1A (Statistiques descriptives) sauvegardée : {chemin_sortie}")

    return table_1a


def generer_table_1b(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Génère le Panel B de la Table 1 : matrice de corrélation linéaire de Pearson."""
    table_1b = df[FACTEURS].corr(method="pearson").round(3)
    table_1b.index.name = "Ticker"
    chemin_sortie = CHEMIN_TABLES / "table_1b.csv"
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    table_1b.to_csv(path_or_buf=chemin_sortie)

    if verbose:
        print(f"Table 1B (Matrice de corrélation linéaire (Pearson)) sauvegardée : {chemin_sortie}")

    return table_1b


def generer_table_1c(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Génère le Panel C de la Table 1 : matrice de corrélation de rang (Spearman)."""
    table_1c = df[FACTEURS].corr(method="spearman").round(3)
    table_1c.index.name = "Ticker"
    chemin_sortie = CHEMIN_TABLES / "table_1c.csv"
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    table_1c.to_csv(path_or_buf=chemin_sortie)

    if verbose:
        print(f"Table 1C (Matrice de corrélation de rang (Spearman)) sauvegardée : {chemin_sortie}")

    return table_1c

def generer_figure_1(df: pd.DataFrame, verbose: bool = True) -> None:
    """Génère la Figure 1 du papier : rendements cumulés des 5 facteurs sur la période OOS et sauvegarde en PNG."""
    import matplotlib.pyplot as plt

    chemin_sortie = Path(__file__).resolve().parents[2] / "results" / "figures" / "figure_1.png"
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    # rendements cumulés : somme cumulée des rendements bruts en %
    df_cumul = (1 + df[FACTEURS] / 100).cumprod() * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    for facteur in FACTEURS:
        ax.plot(df_cumul.index, df_cumul[facteur], label=facteur)

    ax.set_title("Cumulative returns of Fama-French's factors")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement cumulé (base 100)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(chemin_sortie, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"Figure 1 sauvegardée : {chemin_sortie}")


def executer_validation(verbose: bool = True) -> None:
    """Exécute la validation complète : génération des Tables 1A, 1B, 1C."""
    from config.splits import TOTAL_START, TOTAL_END, TRAIN_START, TRAIN_END, TEST_START, TEST_END, OOS_START, OOS_END
    print("ETAPE 01_DATA VALIDATION ================================") if verbose else None

    df_oos = charger_rendements(date_debut=OOS_START, date_fin=OOS_END, log=False, verbose=verbose)

    generer_table_1a(df=df_oos, verbose=verbose)
    generer_table_1b(df=df_oos, verbose=verbose)
    generer_table_1c(df=df_oos, verbose=verbose)

    generer_figure_1(df=df_oos, verbose=verbose)

    print("ETAPE 01_DATA VALIDATION END ============================") if verbose else None


if __name__ == "__main__":
    executer_validation(verbose=True)
    a = True