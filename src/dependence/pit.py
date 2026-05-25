# src/dependency_structure/pit.py

"""
Étape 3b — Transformation PIT (Probability Integral Transform)

On transforme les résidus standardisés z_{i,t} en pseudo-uniformes u_{i,t} ∈ [0,1]
via la fonction de répartition de la loi skewed-t de Hansen (1994) :

    u_{i,t} = F_skt(z_{i,t} ; η_i, λ_i)

Pourquoi la skewed-t et pas la normale ?
    Les résidus financiers ont des queues épaisses et sont asymétriques.
    Utiliser la normale sous-estimerait la probabilité des événements extrêmes,
    ce qui biaiserait la copule.

Les paramètres η (degrés de liberté) et λ (asymétrie) sont estimés par la
librairie arch lors de l'étape GARCH et passés directement ici.

Implémentation :
    On utilise arch.univariate.distribution.SkewStudent qui implémente
    la loi skewed-t de Hansen (1994) — même implémentation que celle utilisée
    lors de l'estimation GARCH, ce qui garantit la cohérence.

Référence :
    Hansen, B. E. (1994). Autoregressive conditional density estimation.
    International Economic Review, 705-730.
"""

import numpy as np
from arch.univariate import SkewStudent


def pit_transform(z: float, eta: float, lam: float) -> float:
    """
    Applique la transformation PIT à un résidu standardisé z.

    Calcule u = F_skt(z ; η, λ) via la CDF de la loi skewed-t de Hansen (1994),
    implémentée dans la librairie arch (cohérence avec l'estimation GARCH).

    Paramètres
    ----------
    z   : float — résidu standardisé issu du GJR-GARCH
    eta : float — degrés de liberté de la skewed-t (η > 2)
                  récupéré depuis result.params['nu'] dans garch.py
    lam : float — paramètre d'asymétrie (-1 < λ < 1)
                  récupéré depuis result.params['lambda'] dans garch.py

    Retourne
    --------
    u : float — pseudo-uniforme ∈ (0, 1)
                clipé à [1e-6, 1-1e-6] pour éviter les problèmes numériques
                dans la copule (qui travaille en log)
    """
    dist = SkewStudent()
    u = dist.cdf(
        resids=np.array([z]),
        parameters=np.array([eta, lam]),  # [η, λ]
    )
    return float(np.clip(u[0], 1e-6, 1 - 1e-6))
