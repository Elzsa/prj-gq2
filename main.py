# main.py

from config import config_dependance

import pandas as pd
from src.dependence.data_loading import load_data, load_data_ext
from src.dependence.garch import (
    rolling_garch,
    save_residuals,
    load_residuals,
    save_uniforms,
    load_uniforms,
)
from src.dependence.dcc import rolling_dcc, save_dcc, load_dcc
from src.dependence.adcc import rolling_adcc, save_adcc, load_adcc
from src.dependence.gas import (
    rolling_gas,
    save_gas,
    load_gas,
    merge_gas_results,
)


def run_replication():
    """Reproduit Zhao et al. (2019) — FF5."""

    config_dependance.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    factors, in_sample, out_sample = load_data()
    print(
        f"[REPRO] {factors.index[0]:%Y-%m} → {factors.index[-1]:%Y-%m}  "
        f"({len(out_sample)} obs OOS)"
    )

    # GARCH + PIT
    residuals_path = config_dependance.RESULTS_DIR / "residuals_garch.parquet"
    uniforms_path = config_dependance.RESULTS_DIR / "uniforms_pit.parquet"
    if residuals_path.exists() and uniforms_path.exists():
        residuals = load_residuals()
        uniforms = load_uniforms()
    else:
        residuals, uniforms = rolling_garch(factors)
        save_residuals(residuals)
        save_uniforms(uniforms)

    # DCC
    dcc_path = config_dependance.RESULTS_DIR / "correlations_dcc.parquet"
    if dcc_path.exists():
        dcc_correlations = load_dcc()
    else:
        dcc_correlations = rolling_dcc(residuals)
        save_dcc(dcc_correlations)

    # ADCC
    adcc_path = config_dependance.RESULTS_DIR / "correlations_adcc.parquet"
    if adcc_path.exists():
        adcc_correlations = load_adcc()
    else:
        adcc_correlations = rolling_adcc(residuals)
        save_adcc(adcc_correlations)

    # GAS
    gas_path = config_dependance.RESULTS_DIR / "correlations_gas.parquet"
    if not gas_path.exists():
        tmp_files = list(
            (config_dependance.RESULTS_DIR / "repro").glob("gas_tmp_*.csv")
        )
        if not tmp_files:
            rolling_gas(
                uniforms,
                start_date=config_dependance.OUT_SAMPLE_START,
                end_date=config_dependance.OUT_SAMPLE_END,
                n_workers=1,
                output_dir=config_dependance.RESULTS_DIR,
            )
        gas_correlations = merge_gas_results(output_dir=config_dependance.RESULTS_DIR)
        save_gas(gas_correlations)

    print("[REPRO] Terminé.")


def run_extension():
    """Extension Momentum — FF5 + MOM."""

    config_dependance.RESULTS_DIR_EXT.mkdir(parents=True, exist_ok=True)

    factors_ext, in_sample_ext, out_sample_ext = load_data_ext()
    print(
        f"[EXT] {factors_ext.index[0]:%Y-%m} → {factors_ext.index[-1]:%Y-%m}  "
        f"({len(out_sample_ext)} obs OOS)"
    )

    # GARCH + PIT
    residuals_ext_path = config_dependance.RESULTS_DIR_EXT / "residuals_garch.parquet"
    uniforms_ext_path = config_dependance.RESULTS_DIR_EXT / "uniforms_pit.parquet"
    if residuals_ext_path.exists() and uniforms_ext_path.exists():
        residuals_ext = pd.read_parquet(residuals_ext_path)
        uniforms_ext = pd.read_parquet(uniforms_ext_path)
    else:
        residuals_ext, uniforms_ext = rolling_garch(factors_ext)
        residuals_ext.to_parquet(residuals_ext_path)
        uniforms_ext.to_parquet(uniforms_ext_path)

    # DCC
    dcc_ext_path = config_dependance.RESULTS_DIR_EXT / "correlations_dcc.parquet"
    if not dcc_ext_path.exists():
        dcc_ext = rolling_dcc(residuals_ext)
        dcc_ext.to_parquet(dcc_ext_path)

    # ADCC
    adcc_ext_path = config_dependance.RESULTS_DIR_EXT / "correlations_adcc.parquet"
    if not adcc_ext_path.exists():
        adcc_ext = rolling_adcc(residuals_ext)
        adcc_ext.to_parquet(adcc_ext_path)

    # GAS
    gas_ext_path = config_dependance.RESULTS_DIR_EXT / "correlations_gas.parquet"
    if not gas_ext_path.exists():
        tmp_files = list(config_dependance.RESULTS_DIR_EXT.glob("gas_tmp_*.csv"))
        if not tmp_files:
            rolling_gas(
                uniforms_ext,
                start_date=config_dependance.OUT_SAMPLE_START,
                end_date=config_dependance.OUT_SAMPLE_END,
                n_workers=1,
                output_dir=config_dependance.RESULTS_DIR_EXT,
            )
        gas_ext = merge_gas_results(output_dir=config_dependance.RESULTS_DIR_EXT)
        gas_ext.to_parquet(gas_ext_path)

    print("[EXT] Terminé.")


if __name__ == "__main__":
    run_replication()
    run_extension()
