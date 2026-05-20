# Réplication et extension d'un papier de recherche

Papier : « Revisiting Fama–French factors' predictability with Bayesian modelling and copula-based portfolio optimization »  
Auteurs : Yang Zhao | Charalampos Stasinakis | Georgios Sermpinis | Filipa Da Silva Fernandes

Langage de programmation : Python
Étudiants : LE NET | MEYER | JULIEN | PAYA

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
git add .                          # pour tout ajouter
git commit -m "Message de commit"
git push origin nom-branche        # la branche sur laquelle push le code
```

## Structure du projet

- `.venv/` : environnement virtuel (ignoré par Git)
- `requirements.txt` : liste des dépendances Python
- `README.md` : ce fichier