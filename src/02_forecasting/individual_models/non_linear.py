# src/02_forecasting/individual_models/nonlinear.py

import sys
from pathlib import Path

# ajout de la racine du projet au sys.path pour permettre les imports absolus
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

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
# decision retenue : 6 lags, coherent avec AR(6) observe en Table 3 du papier Zhao 2019
N_LAGS = 6

# graine aleatoire pour la reproductibilite
SEED = 42

# parametres d'entrainement des reseaux de neurones
# Sermpinis 2017 Table A.2 : iterations optimisees entre 1000 et 100000 sur TEST
# decision retenue : early stopping sur TEST avec patience, max 5000 epochs
EPOCHS_MAX     = 5000
EPOCHS_PATIENCE = 100  # arret si pas d'amelioration sur TEST pendant 100 epochs


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
    """Construit la matrice des inputs X (lags) et la cible y pour les modeles non lineaires."""
    df = pd.DataFrame(index=serie.index)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = serie.shift(lag)
    df["y"] = serie
    return df.dropna()


def preparer_train_test(serie: pd.Series, n_lags: int = N_LAGS) -> tuple:
    """Prepare X_train, y_train, X_test, y_test, idx_train, idx_test depuis une serie de rendements."""
    df = construire_matrice_lags(serie=serie, n_lags=n_lags)

    cols_lags    = [f"lag_{i}" for i in range(1, n_lags + 1)]
    masque_train = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    masque_test  = (df.index >= TEST_START)  & (df.index <= TEST_END)

    X_train   = df.loc[masque_train, cols_lags].values
    y_train   = df.loc[masque_train, "y"].values
    X_test    = df.loc[masque_test,  cols_lags].values
    y_test    = df.loc[masque_test,  "y"].values
    idx_train = df.loc[masque_train].index
    idx_test  = df.loc[masque_test].index

    return X_train, y_train, X_test, y_test, idx_train, idx_test


def concatener_train_test(prev_train: np.ndarray, prev_test: np.ndarray, idx_train: pd.Index, idx_test: pd.Index, nom: str) -> pd.Series:
    """Concatene les previsions TRAIN et TEST en une serie chronologique in-sample complete."""
    return pd.concat([
        pd.Series(data=prev_train, index=idx_train),
        pd.Series(data=prev_test,  index=idx_test)
    ]).rename(nom)


# ==============================================================================
# SECTION 2 : kNN (k-Nearest Neighbours)
# ==============================================================================
# Sermpinis 2017 Appendice A.2.2 :
# "Nearest Neighbors is based on the idea that pieces of time series in the past
#  have patterns which might have resemblance to pieces in the future."
# "The optimal set of parameters is selected based on the highest trading
#  performance in the in-sample period."
# Decision retenue : k optimise par RMSE sur TEST (proxy de la trading performance).
# Sermpinis 2017 : distance euclidienne confirmee.

def prevoir_knn(serie: pd.Series, k: int = 5, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par kNN estime sur TRAIN, previsions TRAIN+TEST."""
    X_train, y_train, X_test, _, idx_train, idx_test = preparer_train_test(serie=serie)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    modele = KNeighborsRegressor(n_neighbors=k, metric="euclidean")
    modele.fit(X=X_train_sc, y=y_train)

    prev_train = modele.predict(X=X_train_sc)
    prev_test  = modele.predict(X=X_test_sc)

    serie_previsions = concatener_train_test(
        prev_train=prev_train, prev_test=prev_test,
        idx_train=idx_train,   idx_test=idx_test,
        nom=f"kNN({k})"
    )

    if verbose:
        print(f"  kNN(k={k}) : {len(serie_previsions)} previsions (TRAIN+TEST)")

    return serie_previsions


# ==============================================================================
# SECTION 3 : MLP (Multilayer Perceptron)
# ==============================================================================
# Sermpinis 2017 Table A.2 :
# - Activation cachee : sigmoide F(z) = 1/(1+e^-z)
# - Activation sortie : somme lineaire F(z) = somme(z)
# - Initialisation : N(0,1)
# - Apprentissage : gradient descent (backpropagation)
# - Iterations : optimisees sur TEST par trial-and-error (1k-100k)
# Decision retenue : early stopping sur TEST avec patience=50, max 5000 epochs.

class _MLP(nn.Module):
    """MLP avec activation sigmoide en couche cachee et sortie lineaire."""

    def __init__(self, n_inputs: int, n_hidden: int):
        """Initialise le MLP avec n_inputs entrees, n_hidden neurones caches, 1 sortie lineaire."""
        super().__init__()
        self.cachee = nn.Linear(in_features=n_inputs, out_features=n_hidden)
        self.sortie = nn.Linear(in_features=n_hidden, out_features=1)
        # initialisation N(0,1) des poids (Sermpinis 2017 Table A.2)
        nn.init.normal_(self.cachee.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.sortie.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.cachee.bias)
        nn.init.zeros_(self.sortie.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant : sigmoide cachee + sortie lineaire."""
        return self.sortie(torch.sigmoid(self.cachee(x)))


def _entrainer_nn_early_stopping(modele: nn.Module, X_train_t: torch.Tensor, y_train_t: torch.Tensor, X_test_t: torch.Tensor, y_test_t: torch.Tensor, epochs_max: int, patience: int) -> nn.Module:
    """Entraine un reseau de neurones avec early stopping sur TEST. Retourne le meilleur modele."""
    # weight_decay=1e-3 : regularisation L2 pour eviter l overfitting sur ~228 observations
    optimizer    = torch.optim.SGD(params=modele.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    critere      = nn.MSELoss()
    meilleur_val = np.inf
    compteur     = 0
    meilleur_etat = None

    modele.train()
    for epoch in range(epochs_max):
        optimizer.zero_grad()
        preds = modele(X_train_t).squeeze()
        perte = critere(preds, y_train_t)
        perte.backward()
        optimizer.step()

        # evaluation sur TEST pour early stopping
        if epoch % 10 == 0:
            modele.eval()
            with torch.no_grad():
                val_preds = modele(X_test_t).squeeze()
                val_loss  = float(critere(val_preds, y_test_t))

            if val_loss < meilleur_val - 1e-7:
                meilleur_val  = val_loss
                meilleur_etat = {k: v.clone() for k, v in modele.state_dict().items()}
                compteur      = 0
            else:
                compteur += 10

            modele.train()
            if compteur >= patience:
                break

    # restaurer le meilleur etat
    if meilleur_etat is not None:
        modele.load_state_dict(meilleur_etat)

    return modele


def prevoir_mlp(serie: pd.Series, n_hidden: int = 5, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par MLP (sigmoide cachee) avec early stopping sur TEST, previsions TRAIN+TEST."""
    X_train, y_train, X_test, y_test, idx_train, idx_test = preparer_train_test(serie=serie)

    scaler_x   = StandardScaler()
    scaler_y   = StandardScaler()
    X_train_sc = scaler_x.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    X_test_sc  = scaler_x.transform(X_test)
    y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    X_train_t = torch.tensor(data=X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(data=y_train_sc, dtype=torch.float32)
    X_test_t  = torch.tensor(data=X_test_sc,  dtype=torch.float32)
    y_test_t  = torch.tensor(data=y_test_sc,  dtype=torch.float32)

    torch.manual_seed(SEED)
    modele = _MLP(n_inputs=X_train.shape[1], n_hidden=n_hidden)
    modele = _entrainer_nn_early_stopping(
        modele=modele, X_train_t=X_train_t, y_train_t=y_train_t,
        X_test_t=X_test_t, y_test_t=y_test_t,
        epochs_max=EPOCHS_MAX, patience=EPOCHS_PATIENCE
    )

    modele.eval()
    with torch.no_grad():
        prev_train_sc = modele(X_train_t).squeeze().numpy()
        prev_test_sc  = modele(X_test_t).squeeze().numpy()

    prev_train = scaler_y.inverse_transform(prev_train_sc.reshape(-1, 1)).ravel()
    prev_test  = scaler_y.inverse_transform(prev_test_sc.reshape(-1, 1)).ravel()

    serie_previsions = concatener_train_test(
        prev_train=prev_train, prev_test=prev_test,
        idx_train=idx_train,   idx_test=idx_test,
        nom=f"MLP(h={n_hidden})"
    )

    if verbose:
        print(f"  MLP(h={n_hidden}) : {len(serie_previsions)} previsions (TRAIN+TEST)")

    return serie_previsions


# ==============================================================================
# SECTION 4 : RNN (Recurrent Neural Network)
# ==============================================================================
# Sermpinis 2017 Appendice A.2.3 et Table A.2 :
# "For an exact specification of recurrent networks, see Elman (1990)."
# - Activation cachee : sigmoide
# - Activation sortie : somme lineaire
# - Initialisation : N(0,1)
# PyTorch nn.RNN utilise tanh par defaut -> on specifie nonlinearity='tanh'
# mais le papier dit sigmoide -> on utilise une implementation manuelle.

class _ElmanRNN(nn.Module):
    """Elman RNN avec activation sigmoide en couche cachee et sortie lineaire."""

    def __init__(self, n_inputs: int, n_hidden: int):
        """Initialise l'Elman RNN avec n_inputs entrees, n_hidden unites cachees, 1 sortie lineaire."""
        super().__init__()
        # poids entree -> cache et cache -> cache
        self.W_ih = nn.Linear(in_features=n_inputs,  out_features=n_hidden)
        self.W_hh = nn.Linear(in_features=n_hidden,  out_features=n_hidden, bias=False)
        self.sortie = nn.Linear(in_features=n_hidden, out_features=1)
        # initialisation N(0,1)
        for param in self.parameters():
            nn.init.normal_(param, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant : sigmoide cachee a chaque pas + sortie lineaire sur le dernier etat cache."""
        # x : (batch, seq_len, n_inputs) ou (batch, n_inputs) si seq_len=1
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, n_inputs)

        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.W_hh.in_features)

        for t in range(x.shape[1]):
            h = torch.sigmoid(self.W_ih(x[:, t, :]) + self.W_hh(h))

        return self.sortie(h)


def prevoir_rnn(serie: pd.Series, n_hidden: int = 5, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par Elman RNN (sigmoide cachee) avec early stopping sur TEST, previsions TRAIN+TEST."""
    X_train, y_train, X_test, y_test, idx_train, idx_test = preparer_train_test(serie=serie)

    scaler_x   = StandardScaler()
    scaler_y   = StandardScaler()
    X_train_sc = scaler_x.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    X_test_sc  = scaler_x.transform(X_test)
    y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    X_train_t = torch.tensor(data=X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(data=y_train_sc, dtype=torch.float32)
    X_test_t  = torch.tensor(data=X_test_sc,  dtype=torch.float32)
    y_test_t  = torch.tensor(data=y_test_sc,  dtype=torch.float32)

    torch.manual_seed(SEED)
    modele = _ElmanRNN(n_inputs=X_train.shape[1], n_hidden=n_hidden)
    modele = _entrainer_nn_early_stopping(
        modele=modele, X_train_t=X_train_t, y_train_t=y_train_t,
        X_test_t=X_test_t, y_test_t=y_test_t,
        epochs_max=EPOCHS_MAX, patience=EPOCHS_PATIENCE
    )

    modele.eval()
    with torch.no_grad():
        prev_train_sc = modele(X_train_t).squeeze().numpy()
        prev_test_sc  = modele(X_test_t).squeeze().numpy()

    prev_train = scaler_y.inverse_transform(prev_train_sc.reshape(-1, 1)).ravel()
    prev_test  = scaler_y.inverse_transform(prev_test_sc.reshape(-1, 1)).ravel()

    serie_previsions = concatener_train_test(
        prev_train=prev_train, prev_test=prev_test,
        idx_train=idx_train,   idx_test=idx_test,
        nom=f"RNN(h={n_hidden})"
    )

    if verbose:
        print(f"  RNN(h={n_hidden}) : {len(serie_previsions)} previsions (TRAIN+TEST)")

    return serie_previsions


# ==============================================================================
# SECTION 5 : HONN (Higher-Order Neural Network)
# ==============================================================================
# Sermpinis 2017 Table A.2 :
# - Activation cachee : sigmoide F(z) = 1/(1+e^-z)
# - Activation sortie : somme lineaire
# - Initialisation : N(0,1)
# "HONNs are able to simulate higher frequency, higher order non-linear data"
# Dunis, Laws & Sermpinis (2011) : HONN enrichit les inputs avec des produits
# croises d'ordre 2, puis applique un reseau standard avec sigmoide cachee.
# Decision retenue : produits croises d'ordre 2 + MLP sigmoide.

def _construire_features_honn(X: np.ndarray, ordre: int = 2) -> np.ndarray:
    """Augmente la matrice X avec les produits croises jusqu'a l'ordre donne."""
    features = [X]
    n        = X.shape[1]
    if ordre >= 2:
        for i in range(n):
            for j in range(i, n):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(features)


class _HONN(nn.Module):
    """HONN : MLP avec features augmentees (produits croises) et activation sigmoide cachee."""

    def __init__(self, n_inputs_augmentes: int, n_hidden: int):
        """Initialise le HONN avec n_inputs_augmentes entrees augmentees, n_hidden neurones caches."""
        super().__init__()
        self.cachee = nn.Linear(in_features=n_inputs_augmentes, out_features=n_hidden)
        self.sortie = nn.Linear(in_features=n_hidden, out_features=1)
        nn.init.normal_(self.cachee.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.sortie.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.cachee.bias)
        nn.init.zeros_(self.sortie.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant : sigmoide cachee + sortie lineaire."""
        return self.sortie(torch.sigmoid(self.cachee(x)))


def prevoir_honn(serie: pd.Series, n_hidden: int = 5, ordre: int = 2, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par HONN (sigmoide cachee, features ordre 2) avec early stopping, previsions TRAIN+TEST."""
    X_train, y_train, X_test, y_test, idx_train, idx_test = preparer_train_test(serie=serie)

    X_train_ho = _construire_features_honn(X=X_train, ordre=ordre)
    X_test_ho  = _construire_features_honn(X=X_test,  ordre=ordre)

    scaler_x   = StandardScaler()
    scaler_y   = StandardScaler()
    X_train_sc = scaler_x.fit_transform(X_train_ho)
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    X_test_sc  = scaler_x.transform(X_test_ho)
    y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    X_train_t = torch.tensor(data=X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(data=y_train_sc, dtype=torch.float32)
    X_test_t  = torch.tensor(data=X_test_sc,  dtype=torch.float32)
    y_test_t  = torch.tensor(data=y_test_sc,  dtype=torch.float32)

    torch.manual_seed(SEED)
    modele = _HONN(n_inputs_augmentes=X_train_sc.shape[1], n_hidden=n_hidden)
    modele = _entrainer_nn_early_stopping(
        modele=modele, X_train_t=X_train_t, y_train_t=y_train_t,
        X_test_t=X_test_t, y_test_t=y_test_t,
        epochs_max=EPOCHS_MAX, patience=EPOCHS_PATIENCE
    )

    modele.eval()
    with torch.no_grad():
        prev_train_sc = modele(X_train_t).squeeze().numpy()
        prev_test_sc  = modele(X_test_t).squeeze().numpy()

    prev_train = scaler_y.inverse_transform(prev_train_sc.reshape(-1, 1)).ravel()
    prev_test  = scaler_y.inverse_transform(prev_test_sc.reshape(-1, 1)).ravel()

    serie_previsions = concatener_train_test(
        prev_train=prev_train, prev_test=prev_test,
        idx_train=idx_train,   idx_test=idx_test,
        nom=f"HONN(ordre={ordre})"
    )

    if verbose:
        print(f"  HONN(ordre={ordre}) : {len(serie_previsions)} previsions (TRAIN+TEST)")

    return serie_previsions


# ==============================================================================
# SECTION 6 : PSN (Psi-Sigma Network)
# ==============================================================================
# Sermpinis 2017 Table A.2 et Appendice A.2.3 :
# "First introduced by Ghosh and Shin (1991)"
# - Activation cachee : produit F(z) = prod(z_psi) pour psi=1..n
# - Activation sortie : sigmoide F(z) = 1/(1+e^-z)  <- DIFFERENT des autres NNs
# - "weights from the hidden to the output layer are fixed to 1"
# - "only the weights from the input to the hidden layer are adjusted"
# - Initialisation : N(0,1)
# Ambiguite : "weights fixed to 1" -> on interprete comme la couche de sortie
# est une somme non ponderee des produits, mais le papier a une sigmoide en sortie.

class _PsiSigmaNetwork(nn.Module):
    """Psi-Sigma Network : produit des entrees ponderees par neurone cache, sortie sigmoide."""

    def __init__(self, n_inputs: int, n_hidden: int):
        """Initialise le PSN avec n_inputs entrees, n_hidden neurones caches."""
        super().__init__()
        # poids input -> cache : seuls parametres entraines (Sermpinis 2017)
        self.poids  = nn.Parameter(torch.empty(n_hidden, n_inputs))
        self.biais  = nn.Parameter(torch.zeros(n_hidden, n_inputs))
        # sortie : poids fixes a 1 selon le papier -> on utilise un Linear non entraine
        self.sortie = nn.Linear(in_features=n_hidden, out_features=1)
        # initialisation N(0,1)
        nn.init.normal_(self.poids, mean=0.0, std=1.0)
        nn.init.normal_(self.sortie.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.sortie.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant : produit des entrees ponderees par neurone, sortie sigmoide."""
        # x : (batch, n_inputs)
        # pour neurone h : prod_i sigmoid(w_hi * x_i + b_hi)
        activations = torch.sigmoid(x.unsqueeze(1) * self.poids + self.biais)  # (batch, n_hidden, n_inputs)
        produits    = activations.prod(dim=2)                                   # (batch, n_hidden)
        return torch.sigmoid(self.sortie(produits))                             # sortie sigmoide


def prevoir_psn(serie: pd.Series, n_hidden: int = 5, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par PSN (sortie sigmoide) avec early stopping sur TEST, previsions TRAIN+TEST."""
    X_train, y_train, X_test, y_test, idx_train, idx_test = preparer_train_test(serie=serie)

    scaler_x   = StandardScaler()
    scaler_y   = StandardScaler()
    X_train_sc = scaler_x.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    X_test_sc  = scaler_x.transform(X_test)
    y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    X_train_t = torch.tensor(data=X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(data=y_train_sc, dtype=torch.float32)
    X_test_t  = torch.tensor(data=X_test_sc,  dtype=torch.float32)
    y_test_t  = torch.tensor(data=y_test_sc,  dtype=torch.float32)

    torch.manual_seed(SEED)
    modele = _PsiSigmaNetwork(n_inputs=X_train.shape[1], n_hidden=n_hidden)
    modele = _entrainer_nn_early_stopping(
        modele=modele, X_train_t=X_train_t, y_train_t=y_train_t,
        X_test_t=X_test_t, y_test_t=y_test_t,
        epochs_max=EPOCHS_MAX, patience=EPOCHS_PATIENCE
    )

    modele.eval()
    with torch.no_grad():
        prev_train_sc = modele(X_train_t).squeeze().numpy()
        prev_test_sc  = modele(X_test_t).squeeze().numpy()

    prev_train = scaler_y.inverse_transform(prev_train_sc.reshape(-1, 1)).ravel()
    prev_test  = scaler_y.inverse_transform(prev_test_sc.reshape(-1, 1)).ravel()

    serie_previsions = concatener_train_test(
        prev_train=prev_train, prev_test=prev_test,
        idx_train=idx_train,   idx_test=idx_test,
        nom=f"PSN(h={n_hidden})"
    )

    if verbose:
        print(f"  PSN(h={n_hidden}) : {len(serie_previsions)} previsions (TRAIN+TEST)")

    return serie_previsions


# ==============================================================================
# SECTION 7 : ARBF-PSO
# ==============================================================================
# Sermpinis 2017 Table A.2 :
# - Activation cachee : gaussienne F(z) = exp(-||z-C||^2 / 2*sigma^2)
# - Activation sortie : somme lineaire
# - Apprentissage : PSO
# Reference : Sermpinis et al. (2013) EJOR pour la description complete.
# Decision retenue : meme implementation PSO que precedemment (centres + sigmas).

def _rbf_output(X: np.ndarray, centres: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Calcule la matrice de sortie RBF gaussienne pour chaque centre."""
    phi = np.zeros((X.shape[0], centres.shape[0]))
    for j in range(centres.shape[0]):
        diff      = X - centres[j]
        phi[:, j] = np.exp(-np.sum(diff ** 2, axis=1) / (2.0 * sigmas[j] ** 2 + 1e-8))
    return phi


def _estimer_poids_rbf(phi_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Estime les poids de sortie du reseau RBF par moindres carres."""
    poids, _, _, _ = np.linalg.lstsq(phi_train, y_train, rcond=None)
    return poids


def prevoir_arbf_pso(serie: pd.Series, n_centres: int = 5, n_particules: int = 20, n_iterations: int = 50, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par ARBF-PSO estime sur TRAIN, previsions TRAIN+TEST."""
    X_train, y_train, X_test, _, idx_train, idx_test = preparer_train_test(serie=serie)

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
    w  = 0.729
    c1 = 1.494
    c2 = 1.494

    def evaluer(pos: np.ndarray) -> float:
        """Evalue la RMSE sur TRAIN pour une position PSO donnee."""
        centres = pos[:n_centres * n_inputs].reshape(n_centres, n_inputs)
        sigmas  = np.abs(pos[n_centres * n_inputs:]) + 0.01
        phi     = _rbf_output(X=X_train_sc, centres=centres, sigmas=sigmas)
        poids   = _estimer_poids_rbf(phi_train=phi, y_train=y_train)
        y_hat   = phi @ poids
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
        r1        = np.random.rand(n_particules, dim_pso)
        r2        = np.random.rand(n_particules, dim_pso)
        vitesses  = w * vitesses + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
        positions = positions + vitesses

    centres_opt = gbest_pos[:n_centres * n_inputs].reshape(n_centres, n_inputs)
    sigmas_opt  = np.abs(gbest_pos[n_centres * n_inputs:]) + 0.01
    phi_train   = _rbf_output(X=X_train_sc, centres=centres_opt, sigmas=sigmas_opt)
    poids_opt   = _estimer_poids_rbf(phi_train=phi_train, y_train=y_train)
    phi_test    = _rbf_output(X=X_test_sc,  centres=centres_opt, sigmas=sigmas_opt)

    prev_train = phi_train @ poids_opt
    prev_test  = phi_test  @ poids_opt

    serie_previsions = concatener_train_test(
        prev_train=prev_train, prev_test=prev_test,
        idx_train=idx_train,   idx_test=idx_test,
        nom=f"ARBF-PSO(k={n_centres})"
    )

    if verbose:
        print(f"  ARBF-PSO(k={n_centres}) : {len(serie_previsions)} previsions, RMSE_train={gbest_val:.6f}")

    return serie_previsions


# ==============================================================================
# SECTION 8 : GP (Genetic Programming)
# ==============================================================================
# Sermpinis 2017 Appendice A.2.4 :
# "GP creates an initial population of models and evolves it using genetic
#  operators (crossover and mutation)."
# "The result is to perform mathematical expressions that best fit to the given input"
# Implementation maison (gplearn incompatible sklearn >= 1.6).
# Representation : arbre d'expression encode en liste prefixe.

_GP_FONCTIONS = [
    ("add",  2, lambda a, b: a + b),
    ("sub",  2, lambda a, b: a - b),
    ("mul",  2, lambda a, b: a * b),
    ("div",  2, lambda a, b: np.divide(a, np.where(np.abs(b) > 1e-6, b, 1.0))),
    ("sqrt", 1, lambda a:    np.sqrt(np.abs(a))),
    ("log",  1, lambda a:    np.log(np.abs(a) + 1e-8)),
]
_GP_N_FONCTIONS    = len(_GP_FONCTIONS)
_GP_PROFONDEUR_MAX = 4


def _gp_arbre_aleatoire(n_terminaux: int, profondeur: int, rng: np.random.Generator) -> list:
    """Genere un arbre GP aleatoire encode en liste prefixe."""
    if profondeur == 0 or rng.random() < 0.3:
        return [("T", rng.integers(0, n_terminaux))]
    idx_fn         = rng.integers(0, _GP_N_FONCTIONS)
    nom, arite, _  = _GP_FONCTIONS[idx_fn]
    noeud          = [("F", idx_fn)]
    for _ in range(arite):
        noeud += _gp_arbre_aleatoire(n_terminaux=n_terminaux, profondeur=profondeur - 1, rng=rng)
    return noeud


def _gp_evaluer(arbre: list, X: np.ndarray) -> tuple:
    """Evalue un arbre GP sur X par parcours prefixe. Retourne (resultat, nb_noeuds_consommes)."""
    if len(arbre) == 0:
        return np.zeros(X.shape[0]), 0
    type_noeud, valeur = arbre[0]
    if type_noeud == "T":
        return X[:, int(valeur) % X.shape[1]].copy(), 1
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
    """Calcule la fitness d'un arbre GP : moins la RMSE. Maximisation."""
    try:
        y_hat, _ = _gp_evaluer(arbre=arbre, X=X)
        if not np.isfinite(y_hat).all():
            return -np.inf
        return -float(np.sqrt(np.mean((y_hat - y) ** 2)))
    except Exception:
        return -np.inf


def _gp_sous_arbre_aleatoire(arbre: list, rng: np.random.Generator) -> tuple:
    """Selectionne un sous-arbre aleatoire et retourne (debut, fin)."""
    taille = len(arbre)
    if taille == 0:
        return 0, 0
    debut = int(rng.integers(0, taille))
    pile  = 1
    fin   = debut
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
    return (parent1[:d1] + parent2[d2:f2] + parent1[f1:])[:64]


def _gp_muter(arbre: list, n_terminaux: int, rng: np.random.Generator) -> list:
    """Mute un arbre en remplacant un sous-arbre par un nouvel arbre aleatoire."""
    d, f = _gp_sous_arbre_aleatoire(arbre=arbre, rng=rng)
    return (arbre[:d] + _gp_arbre_aleatoire(n_terminaux=n_terminaux, profondeur=2, rng=rng) + arbre[f:])[:64]


def prevoir_gp(serie: pd.Series, n_individus: int = 500, n_generations: int = 10, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par GP (implementation maison) estime sur TRAIN, previsions TRAIN+TEST."""
    X_train, y_train, X_test, _, idx_train, idx_test = preparer_train_test(serie=serie)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    n_terminaux = X_train_sc.shape[1]
    rng         = np.random.default_rng(seed=SEED)
    population  = [_gp_arbre_aleatoire(n_terminaux=n_terminaux, profondeur=_GP_PROFONDEUR_MAX, rng=rng)
                   for _ in range(n_individus)]

    meilleur_arbre    = None
    meilleure_fitness = -np.inf

    for _ in range(n_generations):
        scores  = [_gp_fitness(arbre=ind, X=X_train_sc, y=y_train) for ind in population]
        idx_max = int(np.argmax(scores))
        if scores[idx_max] > meilleure_fitness:
            meilleure_fitness = scores[idx_max]
            meilleur_arbre    = population[idx_max].copy()

        nouvelle_pop = [meilleur_arbre]
        while len(nouvelle_pop) < n_individus:
            cand1   = rng.integers(0, n_individus, 3)
            parent1 = population[cand1[int(np.argmax([scores[c] for c in cand1]))]].copy()
            cand2   = rng.integers(0, n_individus, 3)
            parent2 = population[cand2[int(np.argmax([scores[c] for c in cand2]))]].copy()
            enfant  = _gp_croiser(parent1=parent1, parent2=parent2, rng=rng) if rng.random() < 0.9 else _gp_muter(arbre=parent1, n_terminaux=n_terminaux, rng=rng)
            nouvelle_pop.append(enfant)
        population = nouvelle_pop

    if meilleur_arbre is None:
        prev_train = np.zeros(len(idx_train))
        prev_test  = np.zeros(len(idx_test))
    else:
        prev_train, _ = _gp_evaluer(arbre=meilleur_arbre, X=X_train_sc)
        prev_test,  _ = _gp_evaluer(arbre=meilleur_arbre, X=X_test_sc)
        if not np.isfinite(prev_train).all():
            prev_train = np.zeros(len(idx_train))
        if not np.isfinite(prev_test).all():
            prev_test  = np.zeros(len(idx_test))

    serie_previsions = concatener_train_test(
        prev_train=prev_train, prev_test=prev_test,
        idx_train=idx_train,   idx_test=idx_test,
        nom=f"GP(pop={n_individus})"
    )

    if verbose:
        print(f"  GP(pop={n_individus}, gen={n_generations}) : {len(serie_previsions)} previsions, fitness={meilleure_fitness:.6f}")

    return serie_previsions


# ==============================================================================
# SECTION 9 : GEP (Gene Expression Programming)
# ==============================================================================
# Sermpinis 2017 Appendice A.2.4 :
# "GEP is based on symbolic strings of fixed length"
# "Each gene includes a head (detailing symbols specific to functions and terminals)
#  and a tail (only includes terminals)."
# "GEP is considered superior to GP because fitness is established through
#  the genotype and phenotype of an individual."
# Implementation manuelle : chromosome lineaire, Karva language, tournoi, croisement, mutation.

_GEP_FONCTIONS     = {
    "add": (lambda a, b: a + b,                                            2),
    "sub": (lambda a, b: a - b,                                            2),
    "mul": (lambda a, b: a * b,                                            2),
    "div": (lambda a, b: np.divide(a, np.where(np.abs(b) > 1e-6, b, 1.0)), 2),
    "neg": (lambda a:    -a,                                               1),
    "abs": (lambda a:    np.abs(a),                                        1),
}
_GEP_NOM_FONCTIONS = list(_GEP_FONCTIONS.keys())


def _evaluer_chromosome_gep(chromosome: list, X: np.ndarray, h: int) -> np.ndarray:
    """Evalue un chromosome GEP sur X par la methode Karva language."""
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
            pile.append(X[:, (gene - n_fonctions) % n_terminaux].copy())
    if len(pile) == 0:
        return np.zeros(X.shape[0])
    return pile[-1] if isinstance(pile[-1], np.ndarray) else np.full(X.shape[0], float(pile[-1]))


def prevoir_gep(serie: pd.Series, taille_tete: int = 6, n_genes: int = 3, n_pop: int = 50, n_gen: int = 20, verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par GEP simplifie estime sur TRAIN, previsions TRAIN+TEST."""
    X_train, y_train, X_test, _, idx_train, idx_test = preparer_train_test(serie=serie)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    n_fonctions         = len(_GEP_NOM_FONCTIONS)
    n_terminaux         = X_train_sc.shape[1]
    longueur_chromosome = (taille_tete + taille_tete + 1) * n_genes

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
        scores  = [fitness(chrom=c) for c in population]
        idx_max = int(np.argmax(scores))
        if scores[idx_max] > meilleure_fitness:
            meilleure_fitness = scores[idx_max]
            meilleur_chrom    = population[idx_max].copy()

        nouvelle_pop = []
        for _ in range(n_pop):
            cand1   = np.random.choice(n_pop, 3, replace=False)
            parent1 = population[cand1[int(np.argmax([scores[c] for c in cand1]))]].copy()
            cand2   = np.random.choice(n_pop, 3, replace=False)
            parent2 = population[cand2[int(np.argmax([scores[c] for c in cand2]))]].copy()
            point   = np.random.randint(1, longueur_chromosome)
            enfant  = parent1[:point] + parent2[point:]
            for k in range(longueur_chromosome):
                if np.random.rand() < 1.0 / longueur_chromosome:
                    enfant[k] = np.random.randint(0, n_fonctions + n_terminaux)
            nouvelle_pop.append(enfant)
        population = nouvelle_pop

    if meilleur_chrom is None:
        prev_train = np.zeros(len(idx_train))
        prev_test  = np.zeros(len(idx_test))
    else:
        prev_train = _evaluer_chromosome_gep(chromosome=meilleur_chrom, X=X_train_sc, h=taille_tete)
        prev_test  = _evaluer_chromosome_gep(chromosome=meilleur_chrom, X=X_test_sc,  h=taille_tete)
        if not np.isfinite(prev_train).all():
            prev_train = np.zeros(len(idx_train))
        if not np.isfinite(prev_test).all():
            prev_test  = np.zeros(len(idx_test))

    serie_previsions = concatener_train_test(
        prev_train=prev_train, prev_test=prev_test,
        idx_train=idx_train,   idx_test=idx_test,
        nom=f"GEP(h={taille_tete})"
    )

    if verbose:
        print(f"  GEP(h={taille_tete}, n_genes={n_genes}) : {len(serie_previsions)} previsions, fitness={meilleure_fitness:.6f}")

    return serie_previsions


# ==============================================================================
# SECTION 10 : STAR (Smooth Transition AutoRegressive)
# ==============================================================================
# Sermpinis 2017 Appendice A.2.1 et papier Zhao 2019 Section 3.1 :
# "smooth transition autoregressive model" est explicitement dans le pool Zhao 2019.
# Sermpinis 2017 : "two-regime logistic (LSTAR) and exponential (ESTAR) STARs"
# "For both models the orders 1-20 are explored."
# -> 20 LSTAR + 20 ESTAR = 40 modeles STAR au total.
# Zhao 2019 Section 3.1 : "smooth transition autoregressive model" au singulier.
# Decision retenue : 1 seule instance LSTAR et 1 seule instance ESTAR,
# dont l'ordre est selectionne par RMSE minimale sur TEST parmi les ordres 1 a 5.
# Cela reproduit le principe "une instance par famille" du papier.

def _lstar_prevision(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Calcule les previsions LSTAR pour des parametres donnes."""
    # params = [phi1_0, phi1_1, ..., phi1_p, phi2_0, phi2_1, ..., phi2_p, gamma, c]
    p        = (len(params) - 2) // 2
    phi1     = params[:p + 1]
    phi2     = params[p + 1:2 * (p + 1)]
    gamma, c = params[-2], params[-1]
    s        = X[:, 0]  # variable de transition = premier lag
    G        = 1.0 / (1.0 + np.exp(-gamma * (s - c)))
    # prediction regime 1
    y1 = phi1[0] + sum(phi1[j + 1] * X[:, j] for j in range(min(p, X.shape[1])))
    # prediction regime 2
    y2 = phi2[0] + sum(phi2[j + 1] * X[:, j] for j in range(min(p, X.shape[1])))
    return y1 * (1 - G) + y2 * G


def _estar_prevision(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Calcule les previsions ESTAR pour des parametres donnes."""
    p        = (len(params) - 2) // 2
    phi1     = params[:p + 1]
    phi2     = params[p + 1:2 * (p + 1)]
    gamma, c = params[-2], params[-1]
    s        = X[:, 0]
    # fonction de transition exponentielle : G(s) = 1 - exp(-gamma*(s-c)^2)
    G        = 1.0 - np.exp(-np.abs(gamma) * (s - c) ** 2)
    y1 = phi1[0] + sum(phi1[j + 1] * X[:, j] for j in range(min(p, X.shape[1])))
    y2 = phi2[0] + sum(phi2[j + 1] * X[:, j] for j in range(min(p, X.shape[1])))
    return y1 * (1 - G) + y2 * G


def prevoir_star(serie: pd.Series, ordre: int, type_star: str = "LSTAR", verbose: bool = True) -> pd.Series:
    """Prevoit 1 pas en avant par LSTAR ou ESTAR d'ordre donne, estime sur TRAIN, previsions TRAIN+TEST."""
    X_train, y_train, X_test, _, idx_train, idx_test = preparer_train_test(
        serie=serie, n_lags=max(ordre, 1)
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    fn_prev = _lstar_prevision if type_star == "LSTAR" else _estar_prevision

    def objectif(params: np.ndarray) -> float:
        """Critere MSE a minimiser."""
        return float(np.mean((fn_prev(params=params, X=X_train_sc) - y_train) ** 2))

    # initialisation : deux AR(ordre) proches de zero + gamma=1, c=0
    np.random.seed(SEED)
    params_init = np.zeros(2 * (ordre + 1) + 2)
    params_init[-2] = 1.0  # gamma

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(fun=objectif, x0=params_init, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})

    prev_train = fn_prev(params=res.x, X=X_train_sc)
    prev_test  = fn_prev(params=res.x, X=X_test_sc)

    nom = f"{type_star}({ordre})"
    serie_previsions = concatener_train_test(
        prev_train=prev_train, prev_test=prev_test,
        idx_train=idx_train,   idx_test=idx_test,
        nom=nom
    )

    if verbose:
        print(f"  {nom} : {len(serie_previsions)} previsions, convergence={res.success}")

    return serie_previsions


def selectionner_meilleur_ordre_star(serie: pd.Series, type_star: str, ordres: list, verbose: bool = True) -> tuple:
    """Selectionne l'ordre optimal de LSTAR ou ESTAR par RMSE minimale sur TEST. Retourne (ordre_opt, serie_prev)."""
    _, y_train_ref, _, y_test_ref, idx_train_ref, idx_test_ref = preparer_train_test(serie=serie, n_lags=max(ordres))

    meilleur_ordre = ordres[0]
    meilleure_rmse = np.inf
    meilleure_serie = None

    for ordre in ordres:
        try:
            serie_prev = prevoir_star(serie=serie, ordre=ordre, type_star=type_star, verbose=False)
            # RMSE sur TEST uniquement pour la selection
            prev_test = serie_prev.loc[TEST_START:TEST_END].values
            n = min(len(prev_test), len(y_test_ref))
            rmse = float(np.sqrt(np.mean((prev_test[:n] - y_test_ref[:n]) ** 2)))
            if rmse < meilleure_rmse:
                meilleure_rmse = rmse
                meilleur_ordre = ordre
                meilleure_serie = serie_prev
        except Exception:
            continue

    if verbose:
        print(f"  {type_star} : meilleur ordre = {meilleur_ordre} (RMSE_test={meilleure_rmse:.6f})")

    return meilleur_ordre, meilleure_serie


def generer_previsions_star(serie: pd.Series, ordres: list, verbose: bool = True) -> dict:
    """Genere 1 instance LSTAR et 1 instance ESTAR (meilleur ordre par RMSE sur TEST). Retourne un dict nom->Serie."""
    resultats = {}

    for type_star in ["LSTAR", "ESTAR"]:
        if verbose:
            print(f"  -> {type_star} (selection du meilleur ordre parmi {ordres})...")
        ordre_opt, serie_prev = selectionner_meilleur_ordre_star(
            serie=serie, type_star=type_star, ordres=ordres, verbose=verbose
        )
        nom = f"{type_star}({ordre_opt})"
        resultats[nom] = serie_prev

    return resultats


# ==============================================================================
# SECTION 11 : PIPELINE POUR UN FACTEUR
# ==============================================================================
# Registre des modeles non lineaires : (nom, fonction, kwargs)
# STAR est gere separement car il genere plusieurs modeles d'un coup.
_TACHES_NN = [
    ("kNN(5)",        prevoir_knn,      {"k": 5}),
    ("MLP(h=5)",      prevoir_mlp,      {"n_hidden": 5}),
    ("RNN(h=5)",      prevoir_rnn,      {"n_hidden": 5}),
    ("HONN(ordre=2)", prevoir_honn,     {"n_hidden": 5, "ordre": 2}),
    ("PSN(h=5)",      prevoir_psn,      {"n_hidden": 5}),
    ("ARBF-PSO(k=5)", prevoir_arbf_pso, {"n_centres": 5, "n_particules": 20, "n_iterations": 50}),
    ("GP(pop=500)",   prevoir_gp,       {"n_individus": 500, "n_generations": 10}),
    ("GEP(h=6)",      prevoir_gep,      {"taille_tete": 6, "n_genes": 3, "n_pop": 50, "n_gen": 20}),
]

# ordres STAR : 1 a 5 pour LSTAR et ESTAR -> 10 modeles STAR
# Sermpinis 2017 va jusqu'a 20 mais les ordres eleves convergent rarement sur donnees mensuelles
_ORDRES_STAR = [1, 2, 3, 4, 5]


def generer_previsions_nonlineaires_facteur(serie: pd.Series, nom_facteur: str, verbose: bool = True) -> pd.DataFrame:
    """Genere les previsions de tous les modeles non lineaires pour un facteur sur TRAIN+TEST."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Facteur : {nom_facteur}")
        print(f"{'='*60}")

    resultats = {}

    # --- modeles NN et evolutionnaires ---
    for nom, fn, kwargs in _TACHES_NN:
        if verbose:
            print(f"  -> {nom}...")
        resultats[nom] = fn(serie=serie, verbose=False, **kwargs)

    # --- modeles STAR (LSTAR + ESTAR, ordres 1 a 5) ---
    if verbose:
        print(f"  -> STAR (LSTAR+ESTAR ordres {_ORDRES_STAR})...")
    resultats_star = generer_previsions_star(serie=serie, ordres=_ORDRES_STAR, verbose=verbose)
    resultats.update(resultats_star)

    df_complet = pd.DataFrame(data=resultats)

    if verbose:
        n_nans = df_complet.isna().sum().sum()
        print(f"\nTotal : {df_complet.shape[1]} modeles non lineaires, {df_complet.shape[0]} observations")
        print(f"Valeurs manquantes : {n_nans}")

    return df_complet


# ==============================================================================
# SECTION 12 : PIPELINE COMPLET POUR LES 5 FACTEURS
# ==============================================================================

def executer_previsions_nonlineaires(verbose: bool = True) -> dict:
    """Execute les previsions non lineaires pour les 5 facteurs et sauvegarde en CSV multi-index."""
    print("ETAPE 02_FORECASTING INDIVIDUAL NONLINEAR ================") if verbose else None

    df_log = charger_log_rendements(verbose=verbose)

    previsions_par_facteur = {}
    for facteur in FACTEURS:
        serie   = df_log[facteur]
        df_prev = generer_previsions_nonlineaires_facteur(serie=serie, nom_facteur=facteur, verbose=verbose)
        previsions_par_facteur[facteur] = df_prev

    df_global = pd.concat(objs=previsions_par_facteur, axis=1)
    df_global.columns.names = ["facteur", "modele"]

    CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    df_global.to_csv(path_or_buf=CHEMIN_SORTIE, date_format="%Y-%m-%d")

    if verbose:
        print(f"\nPrevisions sauvegardees : {CHEMIN_SORTIE}")
        print(f"Dimensions : {df_global.shape[0]} dates x {df_global.shape[1]} colonnes")
        print("ETAPE 02_FORECASTING INDIVIDUAL NONLINEAR END ============")

    return previsions_par_facteur


# ==============================================================================
# SECTION 13 : CHARGEMENT
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
        exemple = previsions_par_facteur[FACTEURS[0]]
        print(f"Previsions non lineaires chargees : {exemple.shape[0]} dates, {exemple.shape[1]} modeles par facteur")

    return previsions_par_facteur


if __name__ == "__main__":
    previsions = executer_previsions_nonlineaires(verbose=True)
    a = True