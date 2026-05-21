# config/splits.py

from datetime import datetime
# YYYY-MM-DD
TOTAL_START = "1965-01-01" # début du dataset complet
TOTAL_END   = "2017-08-01" # fin du dataset complet

TRAIN_START = "1965-01-01" # début de la période d'entraînement
TRAIN_END   = "1983-12-01" # fin de la période d'entraînement

TEST_START  = "1984-01-01" # début de la période de test
TEST_END    = "1999-12-01" # fin de la période de test

OOS_START   = "2000-01-01" # début de la période hors échantillon Out Of Sample (OOS)
OOS_END     = "2017-08-01" # fin de la période hors échantillon Out Of Sample (OOS)

def valider_splits(verbose: bool = True) -> None:
    """Vérifie que les périodes TRAIN, TEST et OOS sont bien comprises dans la période TOTAL."""
    total_start = datetime.fromisoformat(TOTAL_START)
    total_end   = datetime.fromisoformat(TOTAL_END)

    subsets = {
        "TRAIN": (datetime.fromisoformat(TRAIN_START), datetime.fromisoformat(TRAIN_END)),
        "TEST":  (datetime.fromisoformat(TEST_START),  datetime.fromisoformat(TEST_END)),
        "OOS":   (datetime.fromisoformat(OOS_START),   datetime.fromisoformat(OOS_END)),
    }

    erreurs = []
    for nom, (début, fin) in subsets.items():
        if début < total_start or fin > total_end:
            erreurs.append(
                f"{nom} [{début.date()} - {fin.date()}] "
                f"dépasse TOTAL [{total_start.date()} - {total_end.date()}]"
            )

    if erreurs:
        raise ValueError("Splits invalides :\n" + "\n".join(erreurs))

    if verbose:
        print("Splits valides : TRAIN, TEST et OOS sont bien compris dans TOTAL.")


if __name__ == "__main__":
    valider_splits(verbose=True)