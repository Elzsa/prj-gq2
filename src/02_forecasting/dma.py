# src/02_forecasting/dma.py

import sys
from pathlib import Path

# ajout de la racine du projet au sys.path pour permettre les imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from itertools import combinations

from config.splits import TRAIN_START, TEST_START, TEST_END, OOS_START, OOS_END

# noms des cinq facteurs Fama-French
FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]

# chemins des inputs et outputs
CHEMIN_PCA         = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "pca_components.csv"
CHEMIN_PCA_OBJETS  = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "pca_objets.pkl"
CHEMIN_LINEAR_OOS  = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "individual_predictions_linear_oos.csv"
CHEMIN_NONLINEAR_OOS = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "individual_predictions_nonlinear_oos.csv"
CHEMIN_LOG_RETURNS = Path(__file__).resolve().parents[2] / "data" / "monthly_log_returns.csv"
CHEMIN_SORTIE      = Path(__file__).resolve().parents[2] / "data" / "02_forecasting" / "previsions_dma.csv"

# ==============================================================================
# HYPERPARAMETRES DMA
# ==============================================================================
# Papier Section 3.2.3, Raftery et al. (2010) :
# "In this study, we follow the recommendations of Raftery et al. (2010)
#  and set delta = lambda = 0.99."
DELTA  = 0.99   # facteur d'oubli des poids des modeles
LAMBDA = 0.99   # facteur d'oubli des coefficients

# nombre maximum de composantes PCA utilisees comme inputs de DMA
# le papier dit "This makes DMA impractical with standard computer processing
# when v* is larger than 20." -> on limite a 10 pour avoir U=2^10=1024 modeles
# et rester dans des temps de calcul raisonnables
V_STAR = 10

# initialisation de la variance des coefficients : C_0 = kappa * I
# le papier ne precise pas kappa -> convention standard kappa=1
KAPPA = 1.0


# ==============================================================================
# SECTION 1 : CHARGEMENT DES DONNEES
# ==============================================================================

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


def projeter_previsions_oos(facteur: str, verbose: bool = True) -> pd.DataFrame:
    """Charge les previsions individuelles OOS et les projette sur les axes PCA in-sample."""
    import pickle
    from sklearn.preprocessing import StandardScaler

    with open(CHEMIN_PCA_OBJETS, "rb") as f:
        objets_pca = pickle.load(f)

    df_linear_oos = pd.read_csv(
        filepath_or_buffer=CHEMIN_LINEAR_OOS,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d", header=[0, 1]
    )
    df_nonlinear_oos = pd.read_csv(
        filepath_or_buffer=CHEMIN_NONLINEAR_OOS,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d", header=[0, 1]
    )

    df_oos = pd.concat([df_linear_oos[facteur], df_nonlinear_oos[facteur]], axis=1)
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


# ==============================================================================
# SECTION 2 : CONSTRUCTION DES MODELES CANDIDATS
# ==============================================================================
# Papier Section 3.2.3 : "If we consider a candidate input set u = 1,...,U,
# then the state-space model at time t for the dependent variable y*_t..."
# Chaque modele candidat u est un sous-ensemble des V_STAR composantes PCA.
# On enumere tous les 2^V_STAR sous-ensembles possibles.
# Le modele vide (aucune composante) predit toujours 0.

def construire_modeles_candidats(n_composantes: int, v_star: int) -> list:
    """Construit la liste des 2^v_star sous-ensembles de composantes PCA.
    Chaque element est un tuple d'indices de composantes.
    """
    # on utilise les v_star premieres composantes (celles qui expliquent le plus de variance)
    v_effectif = min(v_star, n_composantes)
    indices    = list(range(v_effectif))

    modeles = []
    for taille in range(0, v_effectif + 1):  # taille 0 = modele vide
        for sous_ensemble in combinations(indices, taille):
            modeles.append(sous_ensemble)

    return modeles


# ==============================================================================
# SECTION 3 : ALGORITHME DMA
# ==============================================================================
# Papier Section 3.2.3, Raftery et al. (2010) :
#
# Equations d'observation et d'etat pour chaque modele u :
#   y*_t = F_t^(u)' * zeta_t^(u) + eps_t^(u),   eps_t^(u) ~ N(0, R_t^(u))
#   zeta_t^(u) = zeta_{t-1}^(u) + eta_t^(u),     eta_t^(u) ~ N(0, V_t^(u))
#
# Mise a jour des coefficients (filtre de Kalman) :
#   zeta_{t|t-1}^(u) = zeta_{t-1|t-1}^(u)
#   C_{t|t-1}^(u)    = (1/lambda) * C_{t-1|t-1}^(u)
#   e_t^(u)          = y*_t - F_t^(u)' * zeta_{t|t-1}^(u)
#   f_t^(u)          = F_t^(u)' * C_{t|t-1}^(u) * F_t^(u) + R_t^(u)
#   K_t^(u)          = C_{t|t-1}^(u) * F_t^(u) / f_t^(u)
#   zeta_{t|t}^(u)   = zeta_{t|t-1}^(u) + K_t^(u) * e_t^(u)
#   C_{t|t}^(u)      = C_{t|t-1}^(u) - K_t^(u) * F_t^(u)' * C_{t|t-1}^(u)
#
# Mise a jour des poids des modeles (facteur d'oubli delta) :
#   omega_{t+1|t,u} = omega_{t|t,u}^delta / sum_l omega_{t|t,l}^delta
#
# Vraisemblance predictive (loi normale) :
#   p(y*_t | M_u, data_{t-1}) = N(y*_t ; F_t^(u)' * zeta_{t|t-1}^(u), f_t^(u))
#
# Mise a jour des poids apres observation :
#   omega_{t|t,u} proportionnel a omega_{t|t-1,u} * p(y*_t | M_u, data_{t-1})
#
# Prevision DMA a chaque date t :
#   y_hat_t = sum_u omega_{t|t-1,u} * F_t^(u)' * zeta_{t|t-1}^(u)

def _log_vraisemblance_normale(y: float, mu: float, sigma2: float) -> float:
    """Calcule le log de la vraisemblance predictive normale N(y; mu, sigma2)."""
    sigma2 = max(sigma2, 1e-10)  # eviter division par zero
    return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (y - mu) ** 2 / sigma2


def executer_dma_facteur(y: np.ndarray, X: np.ndarray, modeles: list, verbose: bool = True) -> np.ndarray:
    """Execute DMA sur une serie y avec features X. Retourne les previsions 1-step ahead.

    Args:
        y       : serie de rendements, shape (T,)
        X       : matrice de composantes PCA, shape (T, n_composantes)
        modeles : liste des sous-ensembles de composantes (tuples d'indices)

    Returns:
        previsions : array de shape (T,), prevision pour chaque date t
    """
    T = len(y)
    U = len(modeles)

    # initialisation des poids egaux pour tous les modeles (Raftery et al. 2010)
    log_poids = np.full(U, -np.log(U))  # log(1/U) pour stabilite numerique

    # initialisation des coefficients et variances pour chaque modele
    # zeta_u : coefficients, shape (U,) avec taille variable selon le modele
    # on stocke sous forme de liste de vecteurs
    zeta   = []
    C_mat  = []
    R_diag = []

    for u, sous_ensemble in enumerate(modeles):
        p = len(sous_ensemble)
        if p == 0:
            zeta.append(np.array([]))
            C_mat.append(np.array([]).reshape(0, 0))
            R_diag.append(float(np.var(y[:max(10, len(y)//10)])))
        else:
            # zeta_0 = 0 (convention standard)
            zeta.append(np.zeros(p))
            # C_0 = kappa * I (convention standard, kappa=1)
            C_mat.append(KAPPA * np.eye(p))
            R_diag.append(float(np.var(y[:max(10, len(y)//10)])))

    previsions = np.zeros(T)

    for t in range(T):
        # ==== ETAPE 1 : prevision avant observation ====
        log_poids_pred = np.zeros(U)
        prev_par_modele = np.zeros(U)

        for u, sous_ensemble in enumerate(modeles):
            p = len(sous_ensemble)

            if p == 0:
                # modele vide : predit 0
                mu_u    = 0.0
                f_u     = R_diag[u]
            else:
                # F_t^(u) : vecteur des composantes selectionnees a la date t
                F_u = X[t, list(sous_ensemble)]

                # prediction : mu = F' * zeta_{t|t-1}
                mu_u = float(F_u @ zeta[u])

                # variance predictive : f = F' * C_{t|t-1} * F + R
                f_u = float(F_u @ C_mat[u] @ F_u) + R_diag[u]

            prev_par_modele[u] = mu_u

            # log-vraisemblance predictive pour ponderation
            if t < T:
                log_poids_pred[u] = log_poids[u] + _log_vraisemblance_normale(
                    y=y[t], mu=mu_u, sigma2=f_u
                )

        # prevision DMA : moyenne ponderee des previsions individuelles
        # on utilise les poids avant observation (omega_{t|t-1})
        poids_norm = log_poids - np.max(log_poids)
        poids_exp  = np.exp(poids_norm)
        poids_exp  = poids_exp / poids_exp.sum()
        previsions[t] = float(poids_exp @ prev_par_modele)

        # ==== ETAPE 2 : mise a jour apres observation ====
        # normalisation des log-vraisemblances pour stabilite
        log_poids_pred = log_poids_pred - np.max(log_poids_pred)
        poids_post = np.exp(log_poids_pred)
        poids_post = poids_post / poids_post.sum()

        # mise a jour des coefficients par filtre de Kalman pour chaque modele
        for u, sous_ensemble in enumerate(modeles):
            p = len(sous_ensemble)
            if p == 0:
                continue

            F_u = X[t, list(sous_ensemble)]

            # propagation de la variance (facteur d'oubli lambda)
            C_pred = C_mat[u] / LAMBDA

            # erreur de prevision
            e_u = y[t] - float(F_u @ zeta[u])

            # variance predictive
            f_u = float(F_u @ C_pred @ F_u) + R_diag[u]

            # gain de Kalman
            K_u = C_pred @ F_u / f_u

            # mise a jour des coefficients
            zeta[u]  = zeta[u] + K_u * e_u
            I = np.eye(len(sous_ensemble))
            A = I - np.outer(K_u, F_u)
            C_mat[u] = A @ C_pred @ A.T + R_diag[u] * np.outer(K_u, K_u)
            # clamping pour eviter la divergence numerique
            valeurs_propres = np.linalg.eigvalsh(C_mat[u])
            if np.any(valeurs_propres < 0) or np.any(np.abs(valeurs_propres) > 1e6):
                C_mat[u] = KAPPA * np.eye(len(sous_ensemble))

            # mise a jour de la variance R par estimation recursive
            R_diag[u] = float(LAMBDA * R_diag[u] + (1 - LAMBDA) * e_u ** 2)
            R_diag[u] = max(R_diag[u], 1e-6)

        # ==== ETAPE 3 : mise a jour des poids avec facteur d'oubli delta ====
        # omega_{t+1|t,u} = omega_{t|t,u}^delta / sum_l omega_{t|t,l}^delta
        log_poids_post = np.log(np.maximum(poids_post, 1e-300))
        log_poids      = DELTA * log_poids_post
        log_poids      = log_poids - np.max(log_poids)  # normalisation numerique

    # clamping des previsions aberrantes : borne a 5*std empirique de y
    seuil = 5.0 * float(np.std(y))
    previsions = np.clip(previsions, -seuil, seuil)

    return previsions


# ==============================================================================
# SECTION 4 : PIPELINE COMPLET POUR UN FACTEUR
# ==============================================================================

def prevoir_dma_facteur(facteur: str, composantes_insample: pd.DataFrame, composantes_oos: pd.DataFrame, df_log: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """Execute DMA sur in-sample puis produit les previsions OOS pour un facteur."""
    if verbose:
        print(f"\n  Facteur : {facteur}")

    # construction des modeles candidats a partir des V_STAR premieres composantes
    n_composantes = composantes_insample.shape[1]
    modeles       = construire_modeles_candidats(n_composantes=n_composantes, v_star=V_STAR)
    v_effectif    = min(V_STAR, n_composantes)

    if verbose:
        print(f"  Modeles candidats : 2^{v_effectif} = {len(modeles)}")

    # concatenation in-sample + OOS pour execution recursive complete
    # DMA apprend en continu : in-sample calibre les poids, OOS continue la recursion
    composantes_completes = pd.concat([composantes_insample, composantes_oos], axis=0)
    composantes_completes = composantes_completes[~composantes_completes.index.duplicated(keep="first")]

    # alignement des vraies valeurs sur la periode complete
    masque_complet = (df_log.index >= composantes_insample.index.min()) & \
                     (df_log.index <= composantes_oos.index.max())
    y_complet = df_log.loc[masque_complet, facteur].values
    X_complet = composantes_completes.iloc[:, :v_effectif].values

    # verification de l'alignement
    n = min(len(y_complet), len(X_complet))
    y_complet = y_complet[:n]
    X_complet = X_complet[:n]
    index_complet = composantes_completes.index[:n]

    if verbose:
        print(f"  Serie complete : {n} observations ({index_complet[0].date()} - {index_complet[-1].date()})")

    # execution DMA sur la serie complete
    previsions_completes = executer_dma_facteur(
        y=y_complet, X=X_complet, modeles=modeles, verbose=False
    )

    # extraction des previsions OOS uniquement
    masque_oos = (index_complet >= pd.Timestamp(OOS_START)) & \
                 (index_complet <= pd.Timestamp(OOS_END))
    previsions_oos = previsions_completes[masque_oos]
    idx_oos        = index_complet[masque_oos]

    serie_previsions = pd.Series(data=previsions_oos, index=idx_oos, name=f"DMA_{facteur}")

    if verbose:
        print(f"  DMA {facteur} : {len(serie_previsions)} previsions OOS")

    return serie_previsions


# ==============================================================================
# SECTION 5 : PIPELINE COMPLET POUR LES 5 FACTEURS
# ==============================================================================

def executer_previsions_dma(verbose: bool = True) -> pd.DataFrame:
    """Execute DMA pour les 5 facteurs. Sauvegarde et retourne les previsions OOS."""
    print("ETAPE 02_FORECASTING DMA =================================") if verbose else None

    composantes_par_facteur = charger_composantes_pca(verbose=verbose)
    df_log                  = charger_log_rendements(verbose=verbose)

    previsions_par_facteur = {}

    for facteur in FACTEURS:
        composantes_oos = projeter_previsions_oos(facteur=facteur, verbose=verbose)
        serie_prev = prevoir_dma_facteur(
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
        print(f"\nPrevisions DMA sauvegardees : {CHEMIN_SORTIE}")
        print(f"Dimensions : {df_previsions.shape[0]} dates OOS x {df_previsions.shape[1]} facteurs")
        print("ETAPE 02_FORECASTING DMA END =============================")

    return df_previsions


# ==============================================================================
# SECTION 6 : CHARGEMENT
# ==============================================================================

def charger_previsions_dma(verbose: bool = True) -> pd.DataFrame:
    """Charge les previsions DMA sauvegardees depuis le CSV."""
    df = pd.read_csv(
        filepath_or_buffer=CHEMIN_SORTIE,
        index_col=0, parse_dates=True, date_format="%Y-%m-%d"
    )

    if verbose:
        print(f"Previsions DMA chargees : {df.shape[0]} dates, {df.shape[1]} facteurs")

    return df


if __name__ == "__main__":
    df_previsions = executer_previsions_dma(verbose=True)
    a = True