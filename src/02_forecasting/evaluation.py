# src/02_forecasting/evaluation.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy import stats

from config.splits import OOS_START, OOS_END

# noms des cinq facteurs Fama-French
FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]

# chemins des inputs
CHEMIN_LOG_RETURNS     = Path(__file__).resolve().parents[2] / "data" / "monthly_log_returns.csv"
CHEMIN_BEST_INDIVIDUEL = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "best_individual.csv"
CHEMIN_NONLINEAR_OOS   = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "individual_predictions_nonlinear_oos.csv"
CHEMIN_LINEAR_OOS      = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "individual_predictions_linear_oos.csv"
CHEMIN_SVR             = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "previsions_svr.csv"
CHEMIN_SCSVR           = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "previsions_sc_svr.csv"
CHEMIN_DMA             = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "previsions_dma.csv"

# chemin de sortie de la Table 4
CHEMIN_TABLE_4 = Path(__file__).resolve().parents[2] / "results" / "tables" / "table_4.csv"


# ==============================================================================
# SECTION 1 : CHARGEMENT DES DONNEES
# ==============================================================================

def charger_vraies_valeurs(verbose: bool = True) -> pd.DataFrame:
    """Charge les vraies valeurs OOS des 5 facteurs."""
    df = pd.read_csv(
        filepath_or_buffer=CHEMIN_LOG_RETURNS,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d"
    )
    df = df[FACTEURS]
    masque = (df.index >= OOS_START) & (df.index <= OOS_END)
    df_oos = df.loc[masque]

    if verbose:
        print(f"Vraies valeurs OOS : {df_oos.shape[0]} observations ({df_oos.index.min().date()} - {df_oos.index.max().date()})")

    return df_oos


def charger_previsions_rw(df_log: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Calcule les previsions RW (Random Walk) : prevision t = valeur reelle t-1.
    Papier Section 3.2 : 'a simple random walk with no trend will also act as naive benchmark.'
    """
    # RW : prevision pour t = rendement observe en t-1
    # on prend la valeur juste avant OOS_START pour la premiere prevision
    masque_complet = df_log.index <= OOS_END
    df_complet     = df_log.loc[masque_complet]
    rw             = df_complet.shift(1)
    masque_oos     = (rw.index >= OOS_START) & (rw.index <= OOS_END)
    df_rw          = rw.loc[masque_oos]

    if verbose:
        print(f"Previsions RW : {df_rw.shape[0]} observations")

    return df_rw


def charger_previsions_best(verbose: bool = True) -> pd.DataFrame:
    """Charge les previsions OOS du meilleur modele individuel par facteur.
    Papier Section 3.1 : 'the best predictor of each case (bold) play the role of our benchmarks.'
    """
    # charger le nom du meilleur modele par facteur
    df_rapport = pd.read_csv(
    filepath_or_buffer=Path(__file__).resolve().parents[2] / "results" / "tables" / "pca_rapport.csv",
    index_col=0
    )

    # charger les previsions OOS nonlineaires et lineaires
    df_nl = pd.read_csv(
        filepath_or_buffer=CHEMIN_NONLINEAR_OOS,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d", header=[0, 1]
    )
    df_lin = pd.read_csv(
        filepath_or_buffer=CHEMIN_LINEAR_OOS,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d", header=[0, 1]
    )

    resultats = {}
    for facteur in FACTEURS:
        nom_best = str(df_rapport.loc[facteur, "meilleur_modele"])

        # chercher d'abord dans nonlineaire puis lineaire
        if nom_best in df_nl[facteur].columns:
            resultats[facteur] = df_nl[facteur][nom_best]
        elif nom_best in df_lin[facteur].columns:
            resultats[facteur] = df_lin[facteur][nom_best]
        else:
            if verbose:
                print(f"  Attention : {nom_best} pour {facteur} non trouve, remplacement par zeros")
            resultats[facteur] = pd.Series(0.0, index=df_nl[facteur].index)

    df_previsions = pd.DataFrame(data=resultats)

    if verbose:
        print(f"Previsions Best : {df_previsions.shape[0]} observations")
        for facteur in FACTEURS:
            nom = str(df_rapport.loc[facteur, "meilleur_modele"])
            print(f"  {facteur} : {nom}")

    return df_previsions


def charger_toutes_previsions(verbose: bool = True) -> dict:
    """Charge toutes les previsions OOS et les vraies valeurs."""
    df_log = pd.read_csv(
        filepath_or_buffer=CHEMIN_LOG_RETURNS,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d"
    )
    df_log = df_log[FACTEURS]

    df_vraies = charger_vraies_valeurs(verbose=verbose)
    df_rw     = charger_previsions_rw(df_log=df_log, verbose=verbose)
    df_best   = charger_previsions_best(verbose=verbose)
    df_svr    = pd.read_csv(CHEMIN_SVR,   index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    df_scsvr  = pd.read_csv(CHEMIN_SCSVR, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    df_dma    = pd.read_csv(CHEMIN_DMA,   index_col=0, parse_dates=True, date_format="%Y-%m-%d")

    return {
        "vraies" : df_vraies,
        "RW"     : df_rw,
        "Best"   : df_best,
        "SVR"    : df_svr,
        "SC-SVR" : df_scsvr,
        "DMA"    : df_dma,
    }


# ==============================================================================
# SECTION 2 : METRIQUES STATISTIQUES
# ==============================================================================
# Papier Section 4 : "The forecasting performance of our models is evaluated
# through four statistics, namely, the root-mean-squared error, the MAE,
# the mean absolute percentage error, and the Theil-U."

def calculer_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE : racine de l'erreur quadratique moyenne."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculer_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE : erreur absolue moyenne."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculer_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE : erreur absolue moyenne en pourcentage.
    On exclut les valeurs ou y_true est proche de zero pour eviter la division par zero.
    """
    masque = np.abs(y_true) > 1e-8
    if masque.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[masque] - y_pred[masque]) / y_true[masque])) * 100)


def calculer_theil_u(y_true: np.ndarray, y_pred: np.ndarray, y_rw: np.ndarray) -> float:
    """Theil-U : RMSE(modele) / RMSE(RW).
    U < 1 signifie que le modele bat le RW.
    """
    rmse_modele = calculer_rmse(y_true=y_true, y_pred=y_pred)
    rmse_rw     = calculer_rmse(y_true=y_true, y_pred=y_rw)
    if rmse_rw < 1e-10:
        return np.nan
    return float(rmse_modele / rmse_rw)


# ==============================================================================
# SECTION 3 : TESTS STATISTIQUES
# ==============================================================================

def test_pt(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Test de Pesaran-Timmermann (1992) : teste si les mouvements directionnels
    de la prevision coincident avec les vrais mouvements.
    Papier note 7 : 'The PT test examines whether the directional movements of
    the real and forecast values are in step with one another.'
    Hypothese nulle : le modele n'a pas de pouvoir predictif directionnel.
    Retourne (statistique PT, p-valeur).
    """
    T = len(y_true)
    if T < 10:
        return np.nan, np.nan

    # proportion de predictions directionnellement correctes
    correct = ((y_true > 0) == (y_pred > 0)).astype(float)
    p_obs   = float(correct.mean())

    # proportions empiriques
    p_y    = float((y_true > 0).mean())
    p_yhat = float((y_pred > 0).mean())

    # proportion attendue sous H0 (independance)
    p_star = p_y * p_yhat + (1 - p_y) * (1 - p_yhat)

    # variance de la statistique
    var_p_obs  = p_star * (1 - p_star) / T
    var_p_star = (2 * p_y - 1) ** 2 * p_yhat * (1 - p_yhat) / T + \
                 (2 * p_yhat - 1) ** 2 * p_y * (1 - p_y) / T

    variance_totale = var_p_obs + var_p_star
    if variance_totale <= 0:
        return np.nan, np.nan

    stat_pt = (p_obs - p_star) / np.sqrt(variance_totale)
    p_val   = 2 * (1 - stats.norm.cdf(abs(stat_pt)))

    return float(stat_pt), float(p_val)


def test_dm(y_true: np.ndarray, y_pred_ref: np.ndarray, y_pred_alt: np.ndarray) -> tuple:
    """Test de Diebold-Mariano (1995) : teste l'egalite de la precision predictive.
    Papier note 7 : 'The DM statistic tests the null hypothesis of equal predictive
    accuracy between two forecasts... a negative realization of the DM value would
    indicate that the DMA forecast is more accurate than the competing forecast.'
    y_pred_ref = previsions DMA (reference), y_pred_alt = previsions du modele concurrent.
    DM < 0 signifie que DMA est plus precis.
    Retourne (statistique DM, p-valeur).
    """
    T = len(y_true)
    if T < 10:
        return np.nan, np.nan

    # differences de pertes quadratiques
    loss_ref = (y_true - y_pred_ref) ** 2
    loss_alt = (y_true - y_pred_alt) ** 2
    d        = loss_ref - loss_alt  # d < 0 signifie que DMA est meilleur

    d_bar = float(np.mean(d))

    # variance de long terme par correction de Harvey et al. (1997)
    gamma_0 = float(np.var(d, ddof=1))
    h       = 1  # horizon de prevision = 1

    # autocorrelation jusqu'a l'ordre h-1
    var_d = gamma_0
    for lag in range(1, h):
        gamma_lag = float(np.cov(d[lag:], d[:-lag])[0, 1])
        var_d    += 2 * gamma_lag

    var_d = max(var_d / T, 1e-10)
    stat_dm = d_bar / np.sqrt(var_d)
    p_val   = 2 * (1 - stats.norm.cdf(abs(stat_dm)))

    return float(stat_dm), float(p_val)


# ==============================================================================
# SECTION 4 : CONSTRUCTION DE LA TABLE 4
# ==============================================================================

def construire_table_4(donnees: dict, verbose: bool = True) -> pd.DataFrame:
    """Construit la Table 4 du papier : performances statistiques OOS.
    Papier Table 4 : MAE, MAPE, RMSE, THEIL-U, statistiques PT et DM par facteur et modele.
    """
    df_vraies = donnees["vraies"]
    df_rw     = donnees["RW"]

    modeles   = ["RW", "Best", "SVR", "SC-SVR", "DMA"]
    metriques = ["MAE", "MAPE", "RMSE", "THEIL-U", "PT_stat", "PT_pval", "DM_stat", "DM_pval"]

    lignes = []

    for facteur in FACTEURS:
        y_true = df_vraies[facteur].values
        y_rw   = df_rw[facteur].values

        # aligner les longueurs
        n = min(len(y_true), len(y_rw))
        y_true = y_true[:n]
        y_rw   = y_rw[:n]

        # previsions DMA pour reference DM
        df_dma  = donnees["DMA"]
        y_dma   = df_dma[facteur].values[:n]

        for nom_modele in modeles:
            df_prev = donnees[nom_modele]

            if facteur in df_prev.columns:
                y_pred = df_prev[facteur].values[:n]
            else:
                y_pred = np.zeros(n)

            # calcul des metriques
            mae     = calculer_mae(y_true=y_true, y_pred=y_pred)
            mape    = calculer_mape(y_true=y_true, y_pred=y_pred)
            rmse    = calculer_rmse(y_true=y_true, y_pred=y_pred)
            theil_u = calculer_theil_u(y_true=y_true, y_pred=y_pred, y_rw=y_rw)

            # test PT
            pt_stat, pt_pval = test_pt(y_true=y_true, y_pred=y_pred)

            # test DM : DMA vs modele concurrent (sauf si modele = DMA)
            if nom_modele == "DMA":
                dm_stat, dm_pval = np.nan, np.nan
            else:
                dm_stat, dm_pval = test_dm(
                    y_true=y_true, y_pred_ref=y_dma, y_pred_alt=y_pred
                )

            lignes.append({
                "Facteur"  : facteur,
                "Modele"   : nom_modele,
                "MAE"      : round(mae,     6),
                "MAPE"     : round(mape,    2),
                "RMSE"     : round(rmse,    6),
                "THEIL-U"  : round(theil_u, 4),
                "PT_stat"  : round(pt_stat, 4) if not np.isnan(pt_stat) else np.nan,
                "PT_pval"  : round(pt_pval, 4) if not np.isnan(pt_pval) else np.nan,
                "DM_stat"  : round(dm_stat, 4) if not np.isnan(dm_stat) else np.nan,
                "DM_pval"  : round(dm_pval, 4) if not np.isnan(dm_pval) else np.nan,
            })

    df_table4 = pd.DataFrame(data=lignes)

    return df_table4

def construire_table_5(df_table4: pd.DataFrame) -> pd.DataFrame:
    """Construit la Table 5 du papier : statistiques PT et DM avec etoiles de significativite.
    Papier Table 5 : PT entre parentheses, DM avec *** p<1%, ** p<5%, * p<10%.
    DMA est la reference pour DM -> pas de DM pour DMA lui-meme.
    """
    def etoiles(pval: float) -> str:
        if np.isnan(pval): return ""
        if pval < 0.01:    return "***"
        if pval < 0.05:    return "**"
        if pval < 0.10:    return "*"
        return ""

    modeles = ["RW", "Best", "SVR", "SC-SVR", "DMA"]
    lignes  = []

    for facteur in FACTEURS:
        ligne_pt = {"Factor": facteur, "Statistic": "PT"}
        ligne_dm = {"Factor": "",      "Statistic": "DM"}

        for modele in modeles:
            sous = df_table4.loc[
                (df_table4["Facteur"] == facteur) & (df_table4["Modele"] == modele)
            ]
            if len(sous) == 0:
                ligne_pt[modele] = ""
                ligne_dm[modele] = ""
                continue

            pt_stat = sous["PT_stat"].values[0]
            pt_pval = sous["PT_pval"].values[0]
            dm_stat = sous["DM_stat"].values[0]
            dm_pval = sous["DM_pval"].values[0]

            # PT : "(valeur)***"
            if not np.isnan(pt_stat):
                ligne_pt[modele] = f"({pt_stat:.2f}){etoiles(pt_pval)}"
            else:
                ligne_pt[modele] = ""

            # DM : "valeur***" ou "-" pour DMA
            if modele == "DMA":
                ligne_dm[modele] = "-"
            elif not np.isnan(dm_stat):
                ligne_dm[modele] = f"{dm_stat:.2f}{etoiles(dm_pval)}"
            else:
                ligne_dm[modele] = ""

        lignes.append(ligne_pt)
        lignes.append(ligne_dm)

    return pd.DataFrame(data=lignes)


def _bootstrap_circulaire(pertes: np.ndarray, taille_bloc: int, n_replications: int, rng: np.random.Generator) -> np.ndarray:
    """Genere n_replications echantillons bootstrap circulaires de la serie de pertes.
    Retourne un array de shape (n_replications, T).
    """
    T        = len(pertes)
    n_blocs  = int(np.ceil(T / taille_bloc))
    resultats = np.zeros((n_replications, T))

    for b in range(n_replications):
        indices = []
        for _ in range(n_blocs):
            debut = rng.integers(0, T)
            for k in range(taille_bloc):
                indices.append((debut + k) % T)
        resultats[b] = pertes[np.array(indices[:T])]

    return resultats


def test_spa(pertes_dict: dict, taille_bloc: int = 15, n_replications: int = 10000, seed: int = 42) -> dict:
    """Test s-SPA de Hsu, Hsu & Kuan (2010) sous critere MAE.
    Pour chaque modele benchmark, teste H0 : le benchmark n'est pas inferieur
    a tous les autres modeles. P-valeur faible -> benchmark inferieur.
    Papier Table 6 : 'Low s-SPA p-values indicate that the benchmark model is
    inferior to at least one of the other models.'
    Retourne un dict modele -> p-valeur.
    """
    rng     = np.random.default_rng(seed=seed)
    modeles = list(pertes_dict.keys())
    T       = len(next(iter(pertes_dict.values())))
    pvaleurs = {}

    for benchmark in modeles:
        pertes_bench = pertes_dict[benchmark]
        # differences de pertes : benchmark - concurrent (positif = benchmark moins bon)
        diffs = []
        for modele in modeles:
            if modele == benchmark:
                continue
            d = pertes_bench - pertes_dict[modele]
            diffs.append(d)

        if len(diffs) == 0:
            pvaleurs[benchmark] = np.nan
            continue

        # statistique SPA : max des moyennes des differences
        moyennes = np.array([d.mean() for d in diffs])
        stat_obs = float(np.max(moyennes))

        # bootstrap pour la distribution nulle
        # sous H0 : centrer chaque difference autour de sa moyenne
        stats_boot = np.zeros(n_replications)
        for b in range(n_replications):
            max_boot = -np.inf
            for d in diffs:
                d_centre = d - d.mean()
                # bloc bootstrap circulaire
                indices  = []
                n_blocs  = int(np.ceil(T / taille_bloc))
                for _ in range(n_blocs):
                    debut = int(rng.integers(0, T))
                    for k in range(taille_bloc):
                        indices.append((debut + k) % T)
                d_boot   = d_centre[np.array(indices[:T])]
                max_boot = max(max_boot, float(d_boot.mean()))
            stats_boot[b] = max_boot

        # p-valeur : proportion des stats bootstrap > stat observee
        pvaleurs[benchmark] = float(np.mean(stats_boot >= stat_obs))

    return pvaleurs


def test_mcs(pertes_dict: dict, taille_bloc: int = 15, n_replications: int = 10000, seed: int = 42, niveau: float = 0.05) -> dict:
    """Test MCS de Hansen, Lunde & Nason (2011) sous critere MAE.
    Elimine iterativement le modele le plus mauvais jusqu'a ce que les restants
    soient statistiquement equivalents au niveau donne.
    Papier Table 6 : 'Low MCS values indicate that the model is not likely to
    belong to the set of the best models.'
    Retourne un dict modele -> p-valeur MCS (1.0 si dans le MCS final).
    """
    rng          = np.random.default_rng(seed=seed)
    modeles_rest = list(pertes_dict.keys())
    T            = len(next(iter(pertes_dict.values())))
    pvaleurs     = {m: 0.0 for m in modeles_rest}

    while len(modeles_rest) > 1:
        M = len(modeles_rest)

        # perte moyenne relative de chaque modele par rapport aux autres
        pertes_mat = np.array([pertes_dict[m] for m in modeles_rest])  # (M, T)
        d_ij       = np.zeros((M, M, T))
        for i in range(M):
            for j in range(M):
                d_ij[i, j] = pertes_mat[i] - pertes_mat[j]

        # perte relative moyenne de chaque modele
        d_i_bar = np.array([d_ij[i].mean(axis=0).mean() for i in range(M)])

        # statistique t_max : modele avec la plus grande perte relative
        variances = np.zeros(M)
        for i in range(M):
            d_i  = d_ij[i].mean(axis=0) - d_i_bar[i]
            # bootstrap pour la variance
            vars_boot = np.zeros(n_replications)
            for b in range(n_replications):
                indices = []
                n_blocs = int(np.ceil(T / taille_bloc))
                for _ in range(n_blocs):
                    debut = int(rng.integers(0, T))
                    for k in range(taille_bloc):
                        indices.append((debut + k) % T)
                d_boot = d_i[np.array(indices[:T])]
                vars_boot[b] = float(d_boot.mean())
            variances[i] = max(float(np.var(vars_boot)), 1e-10)

        t_stats    = d_i_bar / np.sqrt(variances)
        idx_pire   = int(np.argmax(t_stats))
        stat_obs   = float(t_stats[idx_pire])

        # bootstrap pour la p-valeur d'elimination
        stats_boot = np.zeros(n_replications)
        for b in range(n_replications):
            d_i_boot = np.zeros(M)
            for i in range(M):
                indices = []
                n_blocs = int(np.ceil(T / taille_bloc))
                for _ in range(n_blocs):
                    debut = int(rng.integers(0, T))
                    for k in range(taille_bloc):
                        indices.append((debut + k) % T)
                d_centre     = d_ij[i].mean(axis=0) - d_i_bar[i]
                d_i_boot[i]  = float(d_centre[np.array(indices[:T])].mean())
            t_boot = d_i_boot / np.sqrt(variances)
            stats_boot[b] = float(np.max(t_boot))

        pval_elimination = float(np.mean(stats_boot >= stat_obs))

        if pval_elimination < niveau:
            # eliminer le pire modele
            modele_elimine               = modeles_rest[idx_pire]
            pvaleurs[modele_elimine]     = pval_elimination
            modeles_rest.remove(modele_elimine)
        else:
            # tous les modeles restants appartiennent au MCS
            break

    # modeles restants : p-valeur = 1.0 (appartiennent au MCS)
    for m in modeles_rest:
        pvaleurs[m] = 1.0

    return pvaleurs


def construire_table_6(donnees: dict, verbose: bool = True) -> pd.DataFrame:
    """Construit la Table 6 du papier : tests s-SPA et MCS sous critere MAE.
    Papier Table 6 note : 'The table reports the p-values for the s-SPA and MCS
    tests in terms of the MAE criterion.'
    """
    df_vraies = donnees["vraies"]
    modeles   = ["RW", "Best", "SVR", "SC-SVR", "DMA"]
    lignes    = []

    for facteur in FACTEURS:
        if verbose:
            print(f"  Table 6 - {facteur}...")

        y_true = df_vraies[facteur].values
        n      = len(y_true)

        # calcul des pertes MAE par date pour chaque modele
        pertes_dict = {}
        for modele in modeles:
            df_prev = donnees[modele]
            if facteur in df_prev.columns:
                y_pred = df_prev[facteur].values[:n]
            else:
                y_pred = np.zeros(n)
            pertes_dict[modele] = np.abs(y_true - y_pred)

        # tests s-SPA et MCS
        pval_spa = test_spa(pertes_dict=pertes_dict, n_replications=10000)
        pval_mcs = test_mcs(pertes_dict=pertes_dict, n_replications=10000)

        for modele in modeles:
            lignes.append({
                "Facteur" : facteur,
                "Modele"  : modele,
                "s-SPA"   : round(pval_spa.get(modele, np.nan), 4),
                "MCS"     : round(pval_mcs.get(modele, np.nan), 4),
            })

    df_table6 = pd.DataFrame(data=lignes)
    return df_table6

# ==============================================================================
# SECTION 5 : AFFICHAGE ET SAUVEGARDE
# ==============================================================================

def afficher_table_4(df_table4: pd.DataFrame) -> None:
    """Affiche la Table 4 dans le format du papier : Factor, Statistic, RW, Best, SVR, SC-SVR, DMA."""
    print("\n" + "="*80)
    print("TABLE 4 : PERFORMANCES STATISTIQUES OOS (2000-2017)")
    print("="*80)

    metriques = ["MAE", "MAPE", "RMSE", "THEIL-U", "PT_stat", "DM_stat"]
    modeles   = ["RW", "Best", "SVR", "SC-SVR", "DMA"]

    lignes = []
    for facteur in FACTEURS:
        for metrique in metriques:
            ligne = {"Factor": facteur, "Statistic": metrique}
            for modele in modeles:
                val = df_table4.loc[
                    (df_table4["Facteur"] == facteur) & (df_table4["Modele"] == modele),
                    metrique
                ].values
                ligne[modele] = val[0] if len(val) > 0 else np.nan
            lignes.append(ligne)

    df_affichage = pd.DataFrame(data=lignes)
    print(df_affichage.to_string(index=False))


def executer_evaluation(verbose: bool = True) -> pd.DataFrame:
    """Execute l'evaluation complete et sauvegarde la Table 4."""
    print("ETAPE 02_FORECASTING EVALUATION ==========================") if verbose else None

    donnees   = charger_toutes_previsions(verbose=verbose)
    df_table4 = construire_table_4(donnees=donnees, verbose=verbose)

    if verbose:
        afficher_table_4(df_table4=df_table4)

    CHEMIN_TABLE_4.parent.mkdir(parents=True, exist_ok=True)
    # sauvegarde au format du papier : Factor, Statistic, RW, Best, SVR, SC-SVR, DMA
    metriques = ["MAE", "MAPE", "RMSE", "THEIL-U", "PT_stat", "PT_pval", "DM_stat", "DM_pval"]
    modeles   = ["RW", "Best", "SVR", "SC-SVR", "DMA"]
    lignes_sortie = []
    for facteur in FACTEURS:
        for metrique in metriques:
            ligne = {"Factor": facteur, "Statistic": metrique}
            for modele in modeles:
                val = df_table4.loc[
                    (df_table4["Facteur"] == facteur) & (df_table4["Modele"] == modele),
                    metrique
                ].values
                ligne[modele] = val[0] if len(val) > 0 else np.nan
            lignes_sortie.append(ligne)
    df_sortie = pd.DataFrame(data=lignes_sortie)
    df_sortie.to_csv(path_or_buf=CHEMIN_TABLE_4, index=False)

    if verbose:
        print(f"\nTable 4 sauvegardee : {CHEMIN_TABLE_4}")
        print("ETAPE 02_FORECASTING EVALUATION END ======================")

    # construction et sauvegarde Table 5
    df_table5 = construire_table_5(df_table4=df_table4)
    chemin_table5 = Path(__file__).resolve().parents[2] / "results" / "tables" / "table_5.csv"
    df_table5.to_csv(path_or_buf=chemin_table5, index=False)
    if verbose:
        print(f"\nTable 5 :")
        print(df_table5.to_string(index=False))
        print(f"\nTable 5 sauvegardee : {chemin_table5}")

    # construction et sauvegarde Table 6
    if verbose:
        print("\nConstruction Table 6 (s-SPA et MCS, 10000 replications)...")
    df_table6  = construire_table_6(donnees=donnees, verbose=verbose)
    chemin_t6  = Path(__file__).resolve().parents[2] / "results" / "tables" / "table_6.csv"

    # reformater en colonnes Factor, Statistic, RW, Best, SVR, SC-SVR, DMA
    lignes_t6 = []
    for facteur in FACTEURS:
        for stat in ["s-SPA", "MCS"]:
            ligne = {"Factor": facteur, "Statistic": stat}
            for modele in ["RW", "Best", "SVR", "SC-SVR", "DMA"]:
                val = df_table6.loc[
                    (df_table6["Facteur"] == facteur) & (df_table6["Modele"] == modele), stat
                ].values
                ligne[modele] = val[0] if len(val) > 0 else np.nan
            lignes_t6.append(ligne)

    df_t6_formate = pd.DataFrame(data=lignes_t6)
    df_t6_formate.to_csv(path_or_buf=chemin_t6, index=False)
    if verbose:
        print(df_t6_formate.to_string(index=False))
        print(f"\nTable 6 sauvegardee : {chemin_t6}")

    return df_table4


if __name__ == "__main__":
    df_table4 = executer_evaluation(verbose=True)
    a = True