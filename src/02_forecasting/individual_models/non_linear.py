# src/02_forecasting/individual_models/nonlinear.py

import sys
from pathlib import Path

# ajout de la racine du projet au sys.path pour permettre les imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

from config.splits import TRAIN_START, TRAIN_END, TEST_START, TEST_END

# noms des cinq facteurs Fama-French
FACTEURS = ["MKT", "SMB", "HML", "RMW", "CMA"]

# chemin vers les log-rendements produits par preprocess.py
CHEMIN_LOG_RETURNS = Path(__file__).resolve().parents[3] / "data" / "monthly_log_returns.csv"

# chemin de sortie des previsions individuelles non lineaires
CHEMIN_SORTIE = Path(__file__).resolve().parents[3] / "data" / "02_forecasting" / "individual_predictions_nonlinear.csv"

# nombre de lags utilises comme inputs pour tous les modeles non lineaires
# ambiguite du papier : non precise explicitement
# decision retenue : 6 lags, coherent avec les inputs AR observes en Table 3
# (AR(6) apparait pour MKT et RMW)
N_LAGS = 6

# graine aleatoire pour la reproductibilite
SEED = 42


# ==============================================================================
# SECTION 1 : CHARGEMENT ET PREPARATION DES DONNEES
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


def construire_matrice_lags(serie: pd.Series, n_lags: int) -> pd.DataFrame:
    """Construit la matrice des inputs X (lags) et la cible y (valeur a predire) pour les modeles non lineaires."""
    df = pd.DataFrame(index=serie.index)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = serie.shift(lag)
    df["y"] = serie
    return df.dropna()


def preparer_train_test(serie: pd.Series, n_lags: int = N_LAGS) -> tuple:
    """Prepare X_train, y_train, X_test, y_test, index_test depuis une serie de rendements."""
    df = construire_matrice_lags(serie=serie, n_lags=n_lags)

    cols_lags    = [f"lag_{i}" for i in range(1, n_lags + 1)]
    masque_train = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    masque_test  = (df.index >= TEST_START)  & (df.index <= TEST_END)

    X_train  = df.loc[masque_train, cols_lags].values
    y_train  = df.loc[masque_train, "y"].values
    X_test   = df.loc[masque_test,  cols_lags].values
    y_test   = df.loc[masque_test,  "y"].values
    idx_test = df.loc[masque_test].index

    return X_train, y_train, X_test, y_test, idx_test


# ==============================================================================
# SECTION 2 : kNN (k-Nearest Neighbours)
# ==============================================================================
# Papier Section 3.1 : kNN mentionne dans le pool des modeles non lineaires.
# Parametres non precises dans le papier.
# Decision retenue : k = 5, distance euclidienne.
# Ambiguite signalee : le papier ne precise pas k ni la metrique.

def prevoir_knn(serie: pd.Series, k: int = 5, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par kNN estime sur TRAIN, previsions sur TEST."""
    X_train, y_train, X_test, _, idx_test = preparer_train_test(serie=serie)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    modele = KNeighborsRegressor(n_neighbors=k, metric="euclidean")
    modele.fit(X=X_train_sc, y=y_train)
    previsions = modele.predict(X=X_test_sc)

    serie_previsions = pd.Series(data=previsions, index=idx_test, name=f"kNN({k})")

    if verbose:
        print(f"  kNN(k={k}) : {len(serie_previsions)} previsions sur TEST")

    return serie_previsions


# ==============================================================================
# SECTION 3 : MLP (Multilayer Perceptron)
# ==============================================================================
# Papier Section 3.1 : MLP mentionne dans le pool non lineaire.
# Architecture non precisee.
# Decision retenue : 1 couche cachee de 10 neurones, activation tanh, L-BFGS.
# Ambiguite signalee : le papier renvoie a Sermpinis et al. (2017).

def prevoir_mlp(serie: pd.Series, couches_cachees: tuple = (10,), verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par MLP estime sur TRAIN, previsions sur TEST."""
    X_train, y_train, X_test, _, idx_test = preparer_train_test(serie=serie)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modele = MLPRegressor(
            hidden_layer_sizes=couches_cachees,
            activation="tanh",
            solver="lbfgs",
            max_iter=2000,
            random_state=SEED
        )
        modele.fit(X=X_train_sc, y=y_train)

    previsions       = modele.predict(X=X_test_sc)
    serie_previsions = pd.Series(data=previsions, index=idx_test, name=f"MLP{couches_cachees}")

    if verbose:
        print(f"  MLP{couches_cachees} : {len(serie_previsions)} previsions sur TEST")

    return serie_previsions


# ==============================================================================
# SECTION 4 : RNN (Recurrent Neural Network)
# ==============================================================================
# Papier Section 3.1 : RNN mentionne dans le pool non lineaire.
# Decision retenue : Elman RNN, 10 unites cachees, 200 epochs, Adam.
# Ambiguite signalee : le papier ne precise pas si c'est Elman, LSTM ou GRU.

class _ElmanRNN(nn.Module):
    """Reseau Elman RNN simple : 1 couche recurrente + 1 couche lineaire de sortie."""

    def __init__(self, n_inputs: int, n_hidden: int):
        """Initialise le reseau Elman avec n_inputs entrees et n_hidden unites cachees."""
        super().__init__()
        self.rnn    = nn.RNN(input_size=n_inputs, hidden_size=n_hidden, batch_first=True)
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant : RNN puis projection lineaire sur la derniere sortie."""
        out, _ = self.rnn(x)
        return self.linear(out[:, -1, :])


def prevoir_rnn(serie: pd.Series, n_hidden: int = 10, n_epochs: int = 200, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par Elman RNN estime sur TRAIN, previsions sur TEST."""
    X_train, y_train, X_test, _, idx_test = preparer_train_test(serie=serie)

    scaler_x   = StandardScaler()
    scaler_y   = StandardScaler()
    X_train_sc = scaler_x.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    X_test_sc  = scaler_x.transform(X_test)

    # reshape en (batch, seq_len=1, n_features) pour le RNN
    X_train_t = torch.tensor(data=X_train_sc[:, np.newaxis, :], dtype=torch.float32)
    y_train_t = torch.tensor(data=y_train_sc, dtype=torch.float32)
    X_test_t  = torch.tensor(data=X_test_sc[:, np.newaxis, :],  dtype=torch.float32)

    torch.manual_seed(SEED)
    modele    = _ElmanRNN(n_inputs=X_train.shape[1], n_hidden=n_hidden)
    optimizer = torch.optim.Adam(params=modele.parameters(), lr=1e-3)
    critere   = nn.MSELoss()

    modele.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        preds = modele(X_train_t).squeeze()
        perte = critere(preds, y_train_t)
        perte.backward()
        optimizer.step()

    modele.eval()
    with torch.no_grad():
        previsions_sc = modele(X_test_t).squeeze().numpy()

    previsions       = scaler_y.inverse_transform(previsions_sc.reshape(-1, 1)).ravel()
    serie_previsions = pd.Series(data=previsions, index=idx_test, name=f"RNN(h={n_hidden})")

    if verbose:
        print(f"  RNN(h={n_hidden}) : {len(serie_previsions)} previsions sur TEST")

    return serie_previsions


# ==============================================================================
# SECTION 5 : HONN (Higher-Order Neural Network)
# ==============================================================================
# Papier Section 3.1 : HONN mentionne dans le pool non lineaire.
# Enrichit les inputs avec des produits croises d'ordre 2 puis applique un MLP.
# Decision retenue : ordre 2, MLP avec 1 couche de 10 neurones.
# Ambiguite signalee : le papier ne precise pas l'ordre ni l'architecture interne.

def _construire_features_honn(X: np.ndarray, ordre: int = 2) -> np.ndarray:
    """Augmente la matrice X avec les produits croises jusqu'a l'ordre donne."""
    features = [X]
    n        = X.shape[1]

    if ordre >= 2:
        for i in range(n):
            for j in range(i, n):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))

    return np.hstack(features)


def prevoir_honn(serie: pd.Series, ordre: int = 2, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par HONN d'ordre 2 estime sur TRAIN, previsions sur TEST."""
    X_train, y_train, X_test, _, idx_test = preparer_train_test(serie=serie)

    X_train_ho = _construire_features_honn(X=X_train, ordre=ordre)
    X_test_ho  = _construire_features_honn(X=X_test,  ordre=ordre)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_ho)
    X_test_sc  = scaler.transform(X_test_ho)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modele = MLPRegressor(
            hidden_layer_sizes=(10,),
            activation="tanh",
            solver="lbfgs",
            max_iter=2000,
            random_state=SEED
        )
        modele.fit(X=X_train_sc, y=y_train)

    previsions       = modele.predict(X=X_test_sc)
    serie_previsions = pd.Series(data=previsions, index=idx_test, name=f"HONN(ordre={ordre})")

    if verbose:
        print(f"  HONN(ordre={ordre}) : {len(serie_previsions)} previsions sur TEST")

    return serie_previsions


# ==============================================================================
# SECTION 6 : PSN (Psi-Sigma Network)
# ==============================================================================
# Papier Section 3.1 : PSN mentionne dans le pool non lineaire.
# Chaque neurone cache calcule le produit des activations sigmoide de ses entrees.
# Decision retenue : n_hidden = 5, 200 epochs, Adam.
# Ambiguite signalee : architecture exacte non precisee dans le papier.

class _PsiSigmaNetwork(nn.Module):
    """Psi-Sigma Network : produit des activations sigmoide par neurone cache."""

    def __init__(self, n_inputs: int, n_hidden: int):
        """Initialise le PSN avec n_inputs entrees et n_hidden neurones caches."""
        super().__init__()
        self.poids  = nn.Parameter(torch.randn(n_hidden, n_inputs) * 0.1)
        self.biais  = nn.Parameter(torch.zeros(n_hidden, n_inputs))
        self.sortie = nn.Linear(in_features=n_hidden, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant : activation sigmoide puis produit par neurone, puis projection lineaire."""
        # x : (batch, n_inputs)
        activations = torch.sigmoid(x.unsqueeze(1) * self.poids + self.biais)  # (batch, n_hidden, n_inputs)
        produits    = activations.prod(dim=2)                                   # (batch, n_hidden)
        return self.sortie(produits)                                            # (batch, 1)


def prevoir_psn(serie: pd.Series, n_hidden: int = 5, n_epochs: int = 200, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par PSN estime sur TRAIN, previsions sur TEST."""
    X_train, y_train, X_test, _, idx_test = preparer_train_test(serie=serie)

    scaler_x   = StandardScaler()
    scaler_y   = StandardScaler()
    X_train_sc = scaler_x.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    X_test_sc  = scaler_x.transform(X_test)

    X_train_t = torch.tensor(data=X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(data=y_train_sc, dtype=torch.float32)
    X_test_t  = torch.tensor(data=X_test_sc,  dtype=torch.float32)

    torch.manual_seed(SEED)
    modele    = _PsiSigmaNetwork(n_inputs=X_train.shape[1], n_hidden=n_hidden)
    optimizer = torch.optim.Adam(params=modele.parameters(), lr=1e-3)
    critere   = nn.MSELoss()

    modele.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        preds = modele(X_train_t).squeeze()
        perte = critere(preds, y_train_t)
        perte.backward()
        optimizer.step()

    modele.eval()
    with torch.no_grad():
        previsions_sc = modele(X_test_t).squeeze().numpy()

    previsions       = scaler_y.inverse_transform(previsions_sc.reshape(-1, 1)).ravel()
    serie_previsions = pd.Series(data=previsions, index=idx_test, name=f"PSN(h={n_hidden})")

    if verbose:
        print(f"  PSN(h={n_hidden}) : {len(serie_previsions)} previsions sur TEST")

    return serie_previsions


# ==============================================================================
# SECTION 7 : ARBF-PSO (RBF Network optimise par Particle Swarm)
# ==============================================================================
# Papier Section 3.1 : ARBF-PSO mentionne dans le pool non lineaire.
# Reseau RBF dont les centres sont optimises par PSO.
# Decision retenue : n_centres = 5, 20 particules, 50 iterations.
# Ambiguite signalee : le papier ne precise pas n_centres ni les params PSO.

def _rbf_output(X: np.ndarray, centres: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Calcule la matrice de sortie RBF gaussienne pour chaque centre."""
    phi = np.zeros((X.shape[0], centres.shape[0]))
    for j in range(centres.shape[0]):
        diff       = X - centres[j]
        phi[:, j]  = np.exp(-np.sum(diff ** 2, axis=1) / (2.0 * sigmas[j] ** 2 + 1e-8))
    return phi


def _estimer_poids_rbf(phi_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Estime les poids de sortie du reseau RBF par moindres carres."""
    poids, _, _, _ = np.linalg.lstsq(phi_train, y_train, rcond=None)
    return poids


def prevoir_arbf_pso(serie: pd.Series, n_centres: int = 5, n_particules: int = 20, n_iterations: int = 50, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par ARBF-PSO estime sur TRAIN, previsions sur TEST."""
    X_train, y_train, X_test, _, idx_test = preparer_train_test(serie=serie)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    n_inputs = X_train_sc.shape[1]
    dim_pso  = n_centres * n_inputs + n_centres  # centres + sigmas

    np.random.seed(SEED)
    positions  = np.random.randn(n_particules, dim_pso) * 0.5
    vitesses   = np.random.randn(n_particules, dim_pso) * 0.1
    pbest_pos  = positions.copy()
    pbest_val  = np.full(n_particules, np.inf)
    gbest_pos  = None
    gbest_val  = np.inf

    # hyperparametres PSO standard (Clerc & Kennedy 2002)
    w  = 0.729   # inertie
    c1 = 1.494   # acceleration cognitive
    c2 = 1.494   # acceleration sociale

    def evaluer(pos: np.ndarray) -> float:
        """Evalue la RMSE sur TRAIN pour une position PSO donnee."""
        centres  = pos[:n_centres * n_inputs].reshape(n_centres, n_inputs)
        sigmas   = np.abs(pos[n_centres * n_inputs:]) + 0.01
        phi      = _rbf_output(X=X_train_sc, centres=centres, sigmas=sigmas)
        poids    = _estimer_poids_rbf(phi_train=phi, y_train=y_train)
        y_hat    = phi @ poids
        return float(np.sqrt(mean_squared_error(y_true=y_train, y_pred=y_hat)))

    for _ in range(n_iterations):
        for i in range(n_particules):
            val = evaluer(pos=positions[i])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = positions[i].copy()
            if val < gbest_val:
                gbest_val = val
                gbest_pos = positions[i].copy()

        r1         = np.random.rand(n_particules, dim_pso)
        r2         = np.random.rand(n_particules, dim_pso)
        vitesses   = w * vitesses + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
        positions  = positions + vitesses

    centres_opt = gbest_pos[:n_centres * n_inputs].reshape(n_centres, n_inputs)
    sigmas_opt  = np.abs(gbest_pos[n_centres * n_inputs:]) + 0.01
    phi_train   = _rbf_output(X=X_train_sc, centres=centres_opt, sigmas=sigmas_opt)
    poids_opt   = _estimer_poids_rbf(phi_train=phi_train, y_train=y_train)
    phi_test    = _rbf_output(X=X_test_sc,  centres=centres_opt, sigmas=sigmas_opt)
    previsions  = phi_test @ poids_opt

    serie_previsions = pd.Series(data=previsions, index=idx_test, name=f"ARBF-PSO(k={n_centres})")

    if verbose:
        print(f"  ARBF-PSO(n_centres={n_centres}) : {len(serie_previsions)} previsions, RMSE_train={gbest_val:.6f}")

    return serie_previsions


# ==============================================================================
# SECTION 8 : GP (Genetic Programming)
# ==============================================================================
# Papier Section 3.1 : GP mentionne dans le pool non lineaire.
# Decision retenue : 500 individus, 10 generations, fonctions add/sub/mul/div/sqrt/log.
# Ambiguite signalee : le papier ne precise pas les operateurs ni la taille de population.
#
# Implementation maison sans dependance externe (gplearn incompatible sklearn >= 1.6).
# Representation : arbre d'expression encode en liste par parcours en profondeur prefixe.
# Chaque noeud est soit une fonction (noeud interne) soit un terminal (feuille = colonne de X).
# Selection : tournoi de taille 3. Croisement : echange de sous-arbres. Mutation : remplacement.

_GP_FONCTIONS = [
    ("add",  2, lambda a, b: a + b),
    ("sub",  2, lambda a, b: a - b),
    ("mul",  2, lambda a, b: a * b),
    ("div",  2, lambda a, b: np.where(np.abs(b) > 1e-6, a / b, 1.0)),
    ("sqrt", 1, lambda a:    np.sqrt(np.abs(a))),
    ("log",  1, lambda a:    np.log(np.abs(a) + 1e-8)),
]
_GP_N_FONCTIONS  = len(_GP_FONCTIONS)
_GP_PROFONDEUR_MAX = 4   # profondeur max de l'arbre -> au plus 2^4=16 noeuds


def _gp_arbre_aleatoire(n_terminaux: int, profondeur: int, rng: np.random.Generator) -> list:
    """Genere un arbre GP aleatoire encode en liste prefixe (racine en premier)."""
    # a profondeur 0 ou par chance : forcer un terminal
    if profondeur == 0 or rng.random() < 0.3:
        return [("T", rng.integers(0, n_terminaux))]

    # choisir une fonction aleatoire
    idx_fn      = rng.integers(0, _GP_N_FONCTIONS)
    nom, arite, _ = _GP_FONCTIONS[idx_fn]
    noeud       = [("F", idx_fn)]

    for _ in range(arite):
        noeud += _gp_arbre_aleatoire(n_terminaux=n_terminaux, profondeur=profondeur - 1, rng=rng)

    return noeud


def _gp_evaluer(arbre: list, X: np.ndarray) -> tuple:
    """Evalue un arbre GP sur X par parcours prefixe. Retourne (resultat, nb_noeuds_consommes)."""
    if len(arbre) == 0:
        return np.zeros(X.shape[0]), 0

    type_noeud, valeur = arbre[0]

    if type_noeud == "T":
        # terminal : colonne de X
        return X[:, int(valeur) % X.shape[1]].copy(), 1

    # fonction
    _, arite, fn = _GP_FONCTIONS[valeur]
    idx = 1
    args = []
    for _ in range(arite):
        res, consomme = _gp_evaluer(arbre=arbre[idx:], X=X)
        args.append(res)
        idx += consomme

    try:
        resultat = fn(*args)
        resultat = np.where(np.isfinite(resultat), resultat, 0.0)
        resultat = np.clip(resultat, -10.0, 10.0)
    except Exception:
        resultat = np.zeros(X.shape[0])

    return resultat, idx


def _gp_fitness(arbre: list, X: np.ndarray, y: np.ndarray) -> float:
    """Calcule la fitness d'un arbre GP : moins la RMSE sur (X, y). Maximisation."""
    try:
        y_hat, _ = _gp_evaluer(arbre=arbre, X=X)
        if not np.isfinite(y_hat).all():
            return -np.inf
        return -float(np.sqrt(np.mean((y_hat - y) ** 2)))
    except Exception:
        return -np.inf


def _gp_sous_arbre_aleatoire(arbre: list, rng: np.random.Generator) -> tuple:
    """Selectionne un sous-arbre aleatoire et retourne (debut, fin) dans la liste."""
    taille = len(arbre)
    if taille == 0:
        return 0, 0
    debut = int(rng.integers(0, taille))
    # compter la taille du sous-arbre a partir de debut par parcours prefixe
    pile = 1
    fin  = debut
    while pile > 0 and fin < taille:
        type_noeud, valeur = arbre[fin]
        if type_noeud == "F":
            _, arite, _ = _GP_FONCTIONS[valeur]
            pile += arite - 1
        else:
            pile -= 1
        fin += 1
    return debut, fin


def _gp_croiser(parent1: list, parent2: list, rng: np.random.Generator) -> list:
    """Croise deux arbres par echange de sous-arbres aleatoires."""
    d1, f1 = _gp_sous_arbre_aleatoire(arbre=parent1, rng=rng)
    d2, f2 = _gp_sous_arbre_aleatoire(arbre=parent2, rng=rng)
    enfant = parent1[:d1] + parent2[d2:f2] + parent1[f1:]
    # limiter la taille pour eviter les arbres trop profonds
    return enfant[:64]


def _gp_muter(arbre: list, n_terminaux: int, rng: np.random.Generator) -> list:
    """Mute un arbre en remplacant un sous-arbre aleatoire par un nouvel arbre."""
    d, f       = _gp_sous_arbre_aleatoire(arbre=arbre, rng=rng)
    nouvel_arbre = _gp_arbre_aleatoire(n_terminaux=n_terminaux, profondeur=2, rng=rng)
    return (arbre[:d] + nouvel_arbre + arbre[f:])[:64]


def prevoir_gp(serie: pd.Series, n_individus: int = 500, n_generations: int = 10, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par Genetic Programming (implementation maison) estime sur TRAIN, previsions sur TEST."""
    X_train, y_train, X_test, _, idx_test = preparer_train_test(serie=serie)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    n_terminaux = X_train_sc.shape[1]
    rng         = np.random.default_rng(seed=SEED)

    # initialisation de la population
    population = [_gp_arbre_aleatoire(n_terminaux=n_terminaux, profondeur=_GP_PROFONDEUR_MAX, rng=rng)
                  for _ in range(n_individus)]

    meilleur_arbre   = None
    meilleure_fitness = -np.inf

    for _ in range(n_generations):
        scores = [_gp_fitness(arbre=ind, X=X_train_sc, y=y_train) for ind in population]

        # mise a jour du meilleur individu
        idx_max = int(np.argmax(scores))
        if scores[idx_max] > meilleure_fitness:
            meilleure_fitness = scores[idx_max]
            meilleur_arbre    = population[idx_max].copy()

        # construction de la nouvelle population
        nouvelle_pop = [meilleur_arbre]  # elitisme : conserver le meilleur

        while len(nouvelle_pop) < n_individus:
            # selection par tournoi de taille 3
            cand1   = rng.integers(0, n_individus, 3)
            s1      = [scores[c] for c in cand1]
            parent1 = population[cand1[int(np.argmax(s1))]].copy()

            cand2   = rng.integers(0, n_individus, 3)
            s2      = [scores[c] for c in cand2]
            parent2 = population[cand2[int(np.argmax(s2))]].copy()

            # croisement avec proba 0.9, mutation avec proba 0.1
            if rng.random() < 0.9:
                enfant = _gp_croiser(parent1=parent1, parent2=parent2, rng=rng)
            else:
                enfant = _gp_muter(arbre=parent1, n_terminaux=n_terminaux, rng=rng)

            nouvelle_pop.append(enfant)

        population = nouvelle_pop

    # prevision avec le meilleur arbre
    if meilleur_arbre is None:
        previsions = np.zeros(len(idx_test))
    else:
        previsions, _ = _gp_evaluer(arbre=meilleur_arbre, X=X_test_sc)
        if not np.isfinite(previsions).all():
            previsions = np.zeros(len(idx_test))

    serie_previsions = pd.Series(data=previsions, index=idx_test, name=f"GP(pop={n_individus})")

    if verbose:
        print(f"  GP(pop={n_individus}, gen={n_generations}) : {len(serie_previsions)} previsions sur TEST, fitness={meilleure_fitness:.6f}")

    return serie_previsions


# ==============================================================================
# SECTION 9 : GEP (Gene Expression Programming)
# ==============================================================================
# Papier Section 3.1 : GEP mentionne dans le pool non lineaire.
# Implementation manuelle du GEP simplifie (Ferreira 2001) : chromosome lineaire,
# karva language, selection par tournoi, croisement 1 point, mutation.
# Decision retenue : tete h=6, n_genes=3, population=50, generations=20.
# Ambiguite signalee : le papier ne precise pas les hyperparametres GEP.

_GEP_FONCTIONS = {
    "add": (lambda a, b: a + b,                                              2),
    "sub": (lambda a, b: a - b,                                              2),
    "mul": (lambda a, b: a * b,                                              2),
    "div": (lambda a, b: np.where(np.abs(b) > 1e-6, a / b, 1.0),           2),
    "neg": (lambda a:    -a,                                                 1),
    "abs": (lambda a:    np.abs(a),                                          1),
}
_GEP_NOM_FONCTIONS = list(_GEP_FONCTIONS.keys())


def _evaluer_chromosome_gep(chromosome: list, X: np.ndarray, h: int) -> np.ndarray:
    """Evalue un chromosome GEP sur la matrice X par la methode Karva language."""
    n_fonctions = len(_GEP_NOM_FONCTIONS)
    n_terminaux = X.shape[1]
    pile        = []

    for gene in chromosome:
        if gene < n_fonctions:
            nom      = _GEP_NOM_FONCTIONS[gene]
            fn, arite = _GEP_FONCTIONS[nom]
            if arite == 2 and len(pile) >= 2:
                b, a = pile.pop(), pile.pop()
                try:
                    res = np.clip(fn(a, b), -10, 10)
                except Exception:
                    res = np.zeros(X.shape[0])
                pile.append(res)
            elif arite == 1 and len(pile) >= 1:
                a = pile.pop()
                try:
                    res = np.clip(fn(a), -10, 10)
                except Exception:
                    res = np.zeros(X.shape[0])
                pile.append(res)
        else:
            idx_col = (gene - n_fonctions) % n_terminaux
            pile.append(X[:, idx_col].copy())

    if len(pile) == 0:
        return np.zeros(X.shape[0])
    return pile[-1] if isinstance(pile[-1], np.ndarray) else np.full(X.shape[0], float(pile[-1]))


def prevoir_gep(serie: pd.Series, taille_tete: int = 6, n_genes: int = 3, n_pop: int = 50, n_gen: int = 20, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par GEP simplifie estime sur TRAIN, previsions sur TEST."""
    X_train, y_train, X_test, _, idx_test = preparer_train_test(serie=serie)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    n_fonctions          = len(_GEP_NOM_FONCTIONS)
    n_terminaux          = X_train_sc.shape[1]
    longueur_chromosome  = (taille_tete + taille_tete + 1) * n_genes

    np.random.seed(SEED)
    population = [np.random.randint(0, n_fonctions + n_terminaux, longueur_chromosome).tolist() for _ in range(n_pop)]

    def fitness(chrom: list) -> float:
        """Calcule la fitness (moins RMSE) d'un chromosome sur TRAIN."""
        try:
            y_hat = _evaluer_chromosome_gep(chromosome=chrom, X=X_train_sc, h=taille_tete)
            if not np.isfinite(y_hat).all():
                return -np.inf
            return -float(np.sqrt(np.mean((y_hat - y_train) ** 2)))
        except Exception:
            return -np.inf

    meilleur_chrom    = None
    meilleure_fitness = -np.inf

    for _ in range(n_gen):
        scores = [fitness(chrom=c) for c in population]

        idx_max = int(np.argmax(scores))
        if scores[idx_max] > meilleure_fitness:
            meilleure_fitness = scores[idx_max]
            meilleur_chrom    = population[idx_max].copy()

        nouvelle_pop = []
        for _ in range(n_pop):
            # selection par tournoi de taille 3
            cand1    = np.random.choice(n_pop, 3, replace=False)
            parent1  = population[cand1[int(np.argmax([scores[c] for c in cand1]))]].copy()
            cand2    = np.random.choice(n_pop, 3, replace=False)
            parent2  = population[cand2[int(np.argmax([scores[c] for c in cand2]))]].copy()

            # croisement 1 point
            point  = np.random.randint(1, longueur_chromosome)
            enfant = parent1[:point] + parent2[point:]

            # mutation : flip aleatoire avec proba 1/longueur
            for k in range(longueur_chromosome):
                if np.random.rand() < 1.0 / longueur_chromosome:
                    enfant[k] = np.random.randint(0, n_fonctions + n_terminaux)

            nouvelle_pop.append(enfant)
        population = nouvelle_pop

    if meilleur_chrom is None:
        previsions = np.zeros(len(idx_test))
    else:
        previsions = _evaluer_chromosome_gep(chromosome=meilleur_chrom, X=X_test_sc, h=taille_tete)
        if not np.isfinite(previsions).all():
            previsions = np.zeros(len(idx_test))

    serie_previsions = pd.Series(data=previsions, index=idx_test, name=f"GEP(h={taille_tete})")

    if verbose:
        print(f"  GEP(h={taille_tete}, n_genes={n_genes}) : {len(serie_previsions)} previsions, fitness={meilleure_fitness:.6f}")

    return serie_previsions


# ==============================================================================
# SECTION 10 : PIPELINE POUR UN FACTEUR (modeles en parallele)
# ==============================================================================

# registre des taches : (nom_modele, fonction, kwargs)
# structure fixe pour que Parallel puisse distribuer chaque modele independamment
_TACHES_NONLINEAIRES = [
    ("kNN(5)",          prevoir_knn,      {"k": 5}),
    ("MLP(10,)",        prevoir_mlp,      {"couches_cachees": (10,)}),
    ("RNN(h=10)",       prevoir_rnn,      {"n_hidden": 10, "n_epochs": 200}),
    ("HONN(ordre=2)",   prevoir_honn,     {"ordre": 2}),
    ("PSN(h=5)",        prevoir_psn,      {"n_hidden": 5, "n_epochs": 200}),
    ("ARBF-PSO(k=5)",   prevoir_arbf_pso, {"n_centres": 5, "n_particules": 20, "n_iterations": 50}),
    ("GP(pop=500)",     prevoir_gp,       {"n_individus": 500, "n_generations": 10}),
    ("GEP(h=6)",        prevoir_gep,      {"taille_tete": 6, "n_genes": 3, "n_pop": 50, "n_gen": 20}),
]


def _executer_tache_nonlineaire(nom: str, fn, kwargs: dict, serie: pd.Series) -> tuple:
    """Execute une tache non lineaire et retourne (nom, serie_previsions). Compatible Parallel."""
    serie_prev = fn(serie=serie, verbose=False, **kwargs)
    return (nom, serie_prev)


def generer_previsions_nonlineaires_facteur(serie: pd.Series, nom_facteur: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les previsions des 9 familles de modeles non lineaires pour un facteur sur TEST."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Facteur : {nom_facteur}")
        print(f"{'='*60}")

    # execution sequentielle : un modele a la fois pour faciliter le debug
    resultats = {}
    for nom, fn, kwargs in _TACHES_NONLINEAIRES:
        if verbose:
            print(f"  -> {nom}...")
        _, serie_prev = _executer_tache_nonlineaire(nom=nom, fn=fn, kwargs=kwargs, serie=serie)
        resultats[nom] = serie_prev

    df_complet = pd.DataFrame(data={nom: resultats[nom] for nom, _, _ in _TACHES_NONLINEAIRES})

    if verbose:
        n_nans = df_complet.isna().sum().sum()
        print(f"\nTotal : {df_complet.shape[1]} modeles non lineaires, {df_complet.shape[0]} observations")
        print(f"Valeurs manquantes : {n_nans}")

    return df_complet


# ==============================================================================
# SECTION 11 : PIPELINE COMPLET POUR LES 5 FACTEURS
# ==============================================================================

def executer_previsions_nonlineaires(verbose: bool = True) -> dict:
    """Execute les previsions non lineaires pour les 5 facteurs et sauvegarde en CSV multi-index."""
    print("ETAPE 02_FORECASTING INDIVIDUAL NONLINEAR ================") if verbose else None

    df_log = charger_log_rendements(verbose=verbose)

    # execution sequentielle sur les 5 facteurs pour faciliter le debug
    previsions_par_facteur = {}
    for facteur in FACTEURS:
        serie   = df_log[facteur]
        df_prev = generer_previsions_nonlineaires_facteur(serie=serie, nom_facteur=facteur, verbose=verbose)
        previsions_par_facteur[facteur] = df_prev

    # sauvegarde CSV avec multi-index (facteur, modele) en colonnes
    df_global = pd.concat(objs=previsions_par_facteur, axis=1)
    df_global.columns.names = ["facteur", "modele"]

    CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    df_global.to_csv(path_or_buf=CHEMIN_SORTIE, date_format="%Y-%m-%d")

    if verbose:
        print(f"\nPrevisions sauvegardees : {CHEMIN_SORTIE}")
        print(f"Dimensions : {df_global.shape[0]} dates x {df_global.shape[1]} colonnes (5 facteurs x {df_global.shape[1]//5} modeles)")
        print("ETAPE 02_FORECASTING INDIVIDUAL NONLINEAR END ============")

    return previsions_par_facteur


# ==============================================================================
# SECTION 12 : FONCTION DE CHARGEMENT
# ==============================================================================

def charger_previsions_nonlineaires(verbose: bool = True) -> dict:
    """Charge les previsions non lineaires sauvegardees depuis le CSV multi-index."""
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
        print(f"Previsions non lineaires chargees : {df_global.shape[0]} dates, {len(FACTEURS)} facteurs, {df_global.shape[1] // len(FACTEURS)} modeles par facteur")

    return previsions_par_facteur


if __name__ == "__main__":
    previsions = executer_previsions_nonlineaires(verbose=True)
    a = True