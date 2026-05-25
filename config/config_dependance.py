# config.py

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCTION
# ══════════════════════════════════════════════════════════════════════════════

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "F-F_Research_Data_5_Factors_2x3.xlsx"
RESULTS_DIR = BASE_DIR / "results" / "repro"

# ── Facteurs ──────────────────────────────────────────────────────────────────
FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]

# ── Périodes ──────────────────────────────────────────────────────────────────
IN_SAMPLE_START = "1965-01"
IN_SAMPLE_END = "1999-12"
OUT_SAMPLE_START = "2000-01"
OUT_SAMPLE_END = "2025-12"

# ── Fenêtre glissante ─────────────────────────────────────────────────────────
WINDOW_SIZE = 60  # 5 ans en mois

# ── AR + GJR-GARCH ────────────────────────────────────────────────────────────
MAX_AR_LAGS = 6  # ordre AR maximum testé pour la sélection BIC
MAX_GARCH_P = 1
MAX_GARCH_O = 1  # fixe — l'asymétrie GJR est toujours d'ordre 1
MAX_GARCH_Q = 1

# ── Copule ────────────────────────────────────────────────────────────────────
COPULA_CONFIDENCE_LEVELS = [0.95, 0.99]

# ── Portefeuille ──────────────────────────────────────────────────────────────
SHORT_SELLING_RATIO = 0.30  # stratégie 130/30
N_SIMULATIONS = 10000  # Monte Carlo pour CVaR


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION — Momentum (UMD)
# ══════════════════════════════════════════════════════════════════════════════
MOMENTUM_DATA_PATH = BASE_DIR / "data" / "F-F_Momentum_Factor.xlsx"
FACTORS_EXT = FACTORS + ["MOM"]
RESULTS_DIR_EXT = BASE_DIR / "results" / "ext"
