# Réplication et extension d'un papier de recherche

Papier : « Revisiting Fama–French factors' predictability with Bayesian modelling and copula-based portfolio optimization »  
Auteurs : Yang Zhao | Charalampos Stasinakis | Georgios Sermpinis | Filipa Da Silva Fernandes

Langage de programmation : Python  
Étudiants : Arthur LE NET  | Loélia MEYER | Charlène JULIEN | Elsa PAYA

## Installation du projet (Windows et/ou MAC)

1. Cloner le dépôt :
```powershell
git clone https://github.com/Elzsa/prj-gq2.git
```

2. Se déplacer dans le dépôt :
```powershell
cd .\prj-gq2\
```

3. Créer l'environnement virtuel `.venv` :
```powershell
python -m venv .venv
```

4. Activer le `.venv` :

(Windows uniquement) :
```powershell
.venv\Scripts\activate
```

(MAC / Linux uniquement) :
```powershell
source venv/bin/activate
```

> **En cas d'erreur lors de l'activation (Windows) :** exécutez d'abord la commande suivante, puis relancez l'étape 4 :
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

5. Installer les dépendances :
```powershell
pip install -r requirements.txt
```

6. (Optionnel) Désactiver le `.venv` une fois le travail terminé :
```powershell
deactivate
```

## Pour push du code

```powershell
git add nom-fichier1 nom-fichier2  # OU
git add .    # pour tout ajouter
git commit -m "Message de commit"
git push origin nom-branche  # la branche sur laquelle push le code
```

## Structure du projet # TODO REVOIR
```
PRJ-GQ2/
│
├── .venv/ : environnement virtuel (ignoré par Git)
├── .gitignore : fichier des extensions à ignorer par Git
├── README.md : ce fichier
├── requirements.txt : liste des dépendances Python
│
├── data/
│   ├── F-F_Research_Data_5_Factors_2x3.xlsx      # Données brutes French [à récupérer ici](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Research) #TODO VOIR CHARLENE DATA
│   └── log_returns.csv                            # Log-rendements des 5 facteurs
│
├── config/
│   ├── params.py                                      # Hyperparamètres (δ=λ=0.99, β, MC sims...)
│   └── splits.py                                      # Dates train/test/OOS
│
├── src/
│   │
│   ├── 01_data/
│   │   ├── preprocess.py                              # Raw → log-rendements → processed/
│   │   └── validation.py                              # Vérification Table 1 (skewness, kurtosis...)
│   │
│   ├── 02_forecasting/
│   │   ├── individual_models/
│   │   │   ├── linear.py                              # SMA, EMA, AR, ARMA (290 modèles)
│   │   │   └── nonlinear.py                           # MLP, RNN, HONN, GP, GEP, ARBF-PSO
│   │   ├── pca_selection.py                           # ACP → 6-9 composantes par facteur
│   │   ├── svr.py                                     # vSVR + noyau RBF + grid search
│   │   ├── sc_svr.py                                  # Algorithme Sine-Cosine pour calibration SVR
│   │   └── dma.py                                     # Dynamic Model Averaging (δ=λ=0.99)
│   │
│   ├── 03_dependence/
│   │   ├── marginals.py                               # AR(p) + GJR-GARCH + skewed-t Hansen → PIT
│   │   ├── tail_dependence.py                         # UTD/LTD + test asymétrie (Table 7)
│   │   ├── skewed_t_copula.py                         # Copule t asymétrique Demarta-McNeil
│   │   └── gas_dynamics.py                            # Dynamique GAS sur Σₜ
│   │
│   ├── 04_portfolio/
│   │   ├── monte_carlo.py                             # Simulation scénarios depuis copule calibrée
│   │   ├── mean_variance.py                           # Optimisation QP (eq. 11)
│   │   ├── mean_cvar.py                               # Optimisation LP CVaR (eq. 15)
│   │   └── benchmarks.py                              # Stratégies 1/N et Random Walk
│   │
│   ├── 05_evaluation/
│   │   ├── forecast_metrics.py                        # MAE, RMSE, Theil-U, PT, DM, s-SPA, MCS
│   │   └── portfolio_metrics.py                       # Sharpe, Sortino, MDD, Return/CVaR, CDB
│   │
│   └── utils/
│       ├── matrix_utils.py                            # Projection nearest PD matrix
│       └── rolling_window.py                          # Logique fenêtre glissante réutilisable
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_forecasting_results.ipynb
│   ├── 03_copula_analysis.ipynb
│   └── 04_portfolio_results.ipynb
│
├── main.py 
└── results/
    ├── tables/                                        # Tables 1-10 reproduites
    └── figures/                                       # Figure 1 (cumulative returns)
```