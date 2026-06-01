#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12

"""
Build Table 8 and Table 9 from the optimization outputs.

Inputs
------
- results/optimization/returns_*.csv
- results/optimization/weights_*.csv
- data/monthly_log_returns.csv
- data/F-F_Research_Data_5_Factors_2x3.csv

Outputs
-------
- results/tables/table_8.csv
- results/tables/table_8.txt
- results/tables/table_8.png
- results/tables/table_9.csv
- results/tables/table_9.txt
- results/tables/table_9.png
- results/tables/table_10.csv
- results/tables/table_10.txt
- results/tables/table_10.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.splits import OOS_END, OOS_START


ROOT = Path(__file__).resolve().parents[2]
RETURNS_PATH = ROOT / "data" / "monthly_log_returns.csv"
FF_PATH = ROOT / "data" / "F-F_Research_Data_5_Factors_2x3.csv"
DIR_OPTIMIZATION = ROOT / "results" / "optimization"
DIR_OPTIMIZATION_CVAR = ROOT / "results" / "optimization_CVaR"
DIR_TABLES = ROOT / "results" / "tables"

FACTORS = ["MKT", "SMB", "HML", "RMW", "CMA"]

METRIC_COLUMNS = [
    "Annualized return (%)",
    "Sharpe ratio",
    "Sortino ratio",
    "MDD (%)",
    "CDB",
]

TABLE9_METRIC_COLUMNS = [
    "Annualized return (%)",
    "Return/CVaR",
    "Sortino ratio",
    "MDD (%)",
    "CDB",
]

FORECAST_GROUPS = {
    "RW": ["RW-DCC", "RW-ADCC", "RW-GAS"],
    "SC-SVR": ["SC-SVR-DCC", "SC-SVR-ADCC", "SC-SVR-GAS"],
    "DMA": ["DMA-DCC", "DMA-ADCC", "DMA-GAS"],
}

FORECAST_GROUPS_SHORT = {
    "RW": ["RW-DCC-S", "RW-ADCC-S", "RW-GAS-S"],
    "SC-SVR": ["SC-SVR-DCC-S", "SC-SVR-ADCC-S", "SC-SVR-GAS-S"],
    "DMA": ["DMA-DCC-S", "DMA-ADCC-S", "DMA-GAS-S"],
}

FORECAST_GROUPS_CVAR = {
    "RW": ["RW-DCC-SKT", "RW-ADCC-SKT", "RW-GAS-SKT"],
    "SC-SVR": ["SC-SVR-DCC-SKT", "SC-SVR-ADCC-SKT", "SC-SVR-GAS-SKT"],
    "DMA": ["DMA-DCC-SKT", "DMA-ADCC-SKT", "DMA-GAS-SKT"],
}

DEPENDENCE_ORDER = ["DCC", "ADCC", "GAS"]


def _strategy_safe_name(strategy: str) -> str:
    return strategy.replace("/", "_").replace(" ", "_")


def _load_factor_returns_oos() -> pd.DataFrame:
    monthly = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
    )
    mask = (monthly.index >= OOS_START) & (monthly.index <= OOS_END)
    return monthly.loc[mask, FACTORS].copy()


def _load_factor_returns_oos_simple() -> pd.DataFrame:
    monthly_log = _load_factor_returns_oos()
    return np.expm1(monthly_log)


def _load_rf_oos(index: pd.DatetimeIndex) -> pd.Series:
    ff = pd.read_csv(FF_PATH, skiprows=4, index_col=0, dtype=str)
    ff = ff[ff.index.str.match(r"^\d{6}$", na=False)].copy()
    rf = ff[["RF"]].apply(pd.to_numeric, errors="coerce").dropna()
    rf.index = pd.to_datetime(rf.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    rf = np.log(1.0 + rf["RF"] / 100.0)
    return rf.reindex(index).fillna(0.0)


def _load_strategy_outputs(strategy: str) -> tuple[pd.Series, pd.DataFrame]:
    safe_name = _strategy_safe_name(strategy)
    returns_path = DIR_OPTIMIZATION / f"returns_{safe_name}.csv"
    weights_path = DIR_OPTIMIZATION / f"weights_{safe_name}.csv"

    returns = pd.read_csv(
        returns_path,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
    ).iloc[:, 0]
    returns.name = "return"

    weights = pd.read_csv(
        weights_path,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
    )[FACTORS]

    return returns, weights


def _load_cvar_strategy_outputs(
    strategy: str,
    beta: int,
    panel_label: str,
) -> tuple[pd.Series, pd.DataFrame]:
    returns_path = DIR_OPTIMIZATION_CVAR / f"CVaR_{beta}" / f"mean_cvar_{beta}_returns_{panel_label}.csv"
    weights_path = DIR_OPTIMIZATION_CVAR / f"CVaR_{beta}" / f"mean_cvar_{beta}_weights_{panel_label}.csv"

    returns_df = pd.read_csv(
        returns_path,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
    )
    returns_df.index.name = "date"
    returns_df = returns_df.loc[returns_df["model"] == strategy].copy()
    returns = returns_df["portfolio_return"].astype(float)
    returns.name = "return"

    weights_df = pd.read_csv(
        weights_path,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
    )
    weights_df.index.name = "date"
    weights_df = weights_df.loc[weights_df["model"] == strategy, FACTORS].copy().astype(float)

    return returns, weights_df


def _cvar(returns: np.ndarray, confidence_level: float = 0.99) -> float:
    tail_probability = 1.0 - confidence_level
    var_threshold = np.quantile(returns, tail_probability)
    tail = returns[returns <= var_threshold]
    if tail.size == 0:
        return np.nan
    return float(-np.mean(tail))


def _compute_cdb(
    portfolio_returns: np.ndarray,
    mean_weights: np.ndarray,
    factor_returns: np.ndarray,
    q: float = 0.01,
) -> float:
    """
    Conditional Diversification Benefit, appliqué strictement selon l'Appendice E.

    CDB_t(w_t, q) = (CVaR_bar_t^q - CVaR_t^q(w_t)) / (CVaR_bar_t^q - CVaR_underbar_t^q)
    avec :
        CVaR_bar_t^q      = sum_i w_i,t * CVaR_t^q(R_i,t)
        CVaR_underbar_t^q = -F_{p,t}^{-1}(q)

    Aucun retraitement ad hoc des poids n'est appliqué.
    """
    weights = mean_weights.astype(float)

    individual_cvars = np.array(
        [_cvar(factor_returns[:, idx], confidence_level=1.0 - q) for idx in range(factor_returns.shape[1])]
    )

    cvar_bar = float(weights @ individual_cvars)
    cvar_portfolio = _cvar(portfolio_returns, confidence_level=1.0 - q)
    cvar_underbar = float(-np.quantile(portfolio_returns, q))

    denominator = cvar_bar - cvar_underbar
    if denominator <= 0 or np.isnan(denominator):
        return np.nan

    return float((cvar_bar - cvar_portfolio) / denominator)


def _compute_metrics(
    returns: pd.Series,
    rf: pd.Series,
    factor_returns: pd.DataFrame,
    weights: pd.DataFrame | None = None,
) -> dict[str, float]:
    returns = returns.dropna().astype(float)
    if len(returns) < 2:
        return {column: np.nan for column in METRIC_COLUMNS}

    factor_aligned = factor_returns.reindex(returns.index)[FACTORS].astype(float)

    ann_return_pct = float(returns.mean() * 12.0 * 100.0)

    std_returns = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / std_returns * np.sqrt(12.0)) if std_returns > 0 else np.nan

    negative_returns = returns[returns < 0]
    std_negative = float(negative_returns.std(ddof=1)) if len(negative_returns) > 1 else np.nan
    sortino = (
        float(returns.mean() / std_negative * np.sqrt(12.0))
        if pd.notna(std_negative) and std_negative > 0
        else np.nan
    )

    cumulative_value = (1.0 + returns).cumprod()
    running_max = cumulative_value.cummax()
    drawdowns = (cumulative_value - running_max) / running_max
    mdd_pct = float(-drawdowns.min() * 100.0)

    cdb = np.nan
    if weights is not None:
        mean_weights = weights.reindex(returns.index)[FACTORS].mean().values
        cdb = _compute_cdb(
            portfolio_returns=returns.values,
            mean_weights=mean_weights,
            factor_returns=factor_aligned.values,
        )

    return {
        "Annualized return (%)": ann_return_pct,
        "Sharpe ratio": sharpe,
        "Sortino ratio": sortino,
        "MDD (%)": mdd_pct,
        "CDB": cdb,
    }


def _compute_metrics_table_9(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    returns = returns.dropna().astype(float)
    if len(returns) < 2:
        return {column: np.nan for column in TABLE9_METRIC_COLUMNS}

    factor_aligned = factor_returns.reindex(returns.index)[FACTORS].astype(float)
    ann_return_pct = float(returns.mean() * 12.0 * 100.0)
    cvar_val = _cvar(returns.values, confidence_level=confidence_level)
    ret_cvar = float((ann_return_pct / 100.0) / cvar_val) if pd.notna(cvar_val) and cvar_val > 0 else np.nan
    negative_returns = returns[returns < 0]
    std_negative = float(negative_returns.std(ddof=1)) if len(negative_returns) > 1 else np.nan
    sortino = (
        float(returns.mean() / std_negative * np.sqrt(12.0))
        if pd.notna(std_negative) and std_negative > 0
        else np.nan
    )

    cumulative_value = np.exp(returns.cumsum())
    running_max = cumulative_value.cummax()
    drawdowns = (cumulative_value - running_max) / running_max
    mdd_pct = float(-drawdowns.min() * 100.0)

    cdb = np.nan
    if weights is not None:
        mean_weights = weights.reindex(returns.index)[FACTORS].mean().values
        cdb = _compute_cdb(
            portfolio_returns=returns.values,
            mean_weights=mean_weights,
            factor_returns=factor_aligned.values,
        )

    return {
        "Annualized return (%)": ann_return_pct,
        "Return/CVaR": ret_cvar,
        "Sortino ratio": sortino,
        "MDD (%)": mdd_pct,
        "CDB": cdb,
    }


def _make_row(panel: str, model: str, metrics: dict[str, float]) -> dict[str, object]:
    row: dict[str, object] = {"Panel": panel, "Models": model}
    row.update(metrics)
    return row


def _average_rows(
    panel: str,
    model: str,
    rows: list[dict[str, object]],
    metric_columns: list[str] = METRIC_COLUMNS,
) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    metrics = {column: float(frame[column].mean()) for column in metric_columns}
    return _make_row(panel=panel, model=model, metrics=metrics)


def _strategy_matches_dependence(strategy: str, dependence: str, dependence_suffix: str) -> bool:
    suffix = f"-{dependence}{dependence_suffix}"
    return strategy.endswith(suffix)


def build_panel_a(factor_returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for factor in FACTORS:
        metrics = _compute_metrics(
            returns=factor_returns[factor],
            rf=rf,
            factor_returns=factor_returns,
            weights=None,
        )
        metrics["CDB"] = np.nan
        rows.append(_make_row(panel="Panel A", model=factor, metrics=metrics))

    returns_1n, weights_1n = _load_strategy_outputs("1/N")
    metrics_1n = _compute_metrics(
        returns=returns_1n,
        rf=rf,
        factor_returns=factor_returns,
        weights=weights_1n,
    )
    rows.append(_make_row(panel="Panel A", model="1/N", metrics=metrics_1n))

    return pd.DataFrame(rows)


def build_panel_b_or_c(
    factor_returns: pd.DataFrame,
    rf: pd.Series,
    forecast_groups: dict[str, list[str]],
    panel_name: str,
    total_average_label: str,
    dependence_suffix: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []

    for strategies in forecast_groups.values():
        group_rows: list[dict[str, object]] = []
        for strategy in strategies:
            returns, weights = _load_strategy_outputs(strategy)
            metrics = _compute_metrics(
                returns=returns,
                rf=rf,
                factor_returns=factor_returns,
                weights=weights,
            )
            row = _make_row(panel=panel_name, model=strategy, metrics=metrics)
            rows.append(row)
            group_rows.append(row)
            base_rows.append(row)

        rows.append(_average_rows(panel=panel_name, model="Average", rows=group_rows))

    rows.append(_average_rows(panel=panel_name, model=total_average_label, rows=base_rows))

    for dependence in DEPENDENCE_ORDER:
        dependence_rows = [
            row
            for row in base_rows
            if _strategy_matches_dependence(
                strategy=str(row["Models"]),
                dependence=dependence,
                dependence_suffix=dependence_suffix,
            )
        ]
        if dependence_rows:
            label = f"{dependence}{dependence_suffix} Average"
            rows.append(_average_rows(panel=panel_name, model=label, rows=dependence_rows))

    return pd.DataFrame(rows)


def build_table_8() -> pd.DataFrame:
    factor_returns = _load_factor_returns_oos_simple()
    rf = _load_rf_oos(factor_returns.index)

    panel_a = build_panel_a(factor_returns=factor_returns, rf=rf)
    panel_b = build_panel_b_or_c(
        factor_returns=factor_returns,
        rf=rf,
        forecast_groups=FORECAST_GROUPS,
        panel_name="Panel B",
        total_average_label="Total Average",
    )
    panel_c = build_panel_b_or_c(
        factor_returns=factor_returns,
        rf=rf,
        forecast_groups=FORECAST_GROUPS_SHORT,
        panel_name="Panel C",
        total_average_label="Total Average - S",
        dependence_suffix="-S",
    )

    table = pd.concat([panel_a, panel_b, panel_c], ignore_index=True)
    table[METRIC_COLUMNS] = table[METRIC_COLUMNS].round(3)
    return table


def build_panel_a_table_9(
    factor_returns: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for factor in FACTORS:
        metrics = _compute_metrics_table_9(
            returns=factor_returns[factor],
            factor_returns=factor_returns,
            weights=None,
            confidence_level=confidence_level,
        )
        metrics["CDB"] = np.nan
        rows.append(_make_row(panel="Panel A", model=factor, metrics=metrics))

    weights_1n = pd.read_csv(
        DIR_OPTIMIZATION_CVAR / "mean_cvar_weights_1N.csv",
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
    )[FACTORS].astype(float)
    returns_1n = pd.read_csv(
        DIR_OPTIMIZATION_CVAR / "mean_cvar_returns_1N.csv",
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
    ).iloc[:, 0].astype(float)
    metrics_1n = _compute_metrics_table_9(
        returns=returns_1n,
        factor_returns=factor_returns,
        weights=weights_1n,
        confidence_level=confidence_level,
    )
    rows.append(_make_row(panel="Panel A", model="1/N", metrics=metrics_1n))
    return pd.DataFrame(rows)


def build_panel_b_or_c_table_9(
    factor_returns: pd.DataFrame,
    forecast_groups: dict[str, list[str]],
    panel_name: str,
    panel_label: str,
    beta: int = 95,
    model_suffix: str = "",
    total_average_label: str = "Total Average",
    dependence_suffix: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    confidence_level = beta / 100.0

    for strategies in forecast_groups.values():
        group_rows: list[dict[str, object]] = []
        for strategy in strategies:
            returns, weights = _load_cvar_strategy_outputs(strategy=strategy, beta=beta, panel_label=panel_label)
            metrics = _compute_metrics_table_9(
                returns=returns,
                factor_returns=factor_returns,
                weights=weights,
                confidence_level=confidence_level,
            )
            row = _make_row(panel=panel_name, model=f"{strategy}{model_suffix}", metrics=metrics)
            rows.append(row)
            group_rows.append(row)
            base_rows.append(row)

        rows.append(
            _average_rows(
                panel=panel_name,
                model="Average",
                rows=group_rows,
                metric_columns=TABLE9_METRIC_COLUMNS,
            )
        )

    rows.append(
        _average_rows(
            panel=panel_name,
            model=total_average_label,
            rows=base_rows,
            metric_columns=TABLE9_METRIC_COLUMNS,
        )
    )

    for dependence in DEPENDENCE_ORDER:
        dependence_rows = [row for row in base_rows if f"-{dependence}-" in str(row["Models"])]
        if dependence_rows:
            rows.append(
                _average_rows(
                    panel=panel_name,
                    model=f"{dependence}{dependence_suffix} Average",
                    rows=dependence_rows,
                    metric_columns=TABLE9_METRIC_COLUMNS,
                )
            )

    return pd.DataFrame(rows)


def build_table_9(beta: int = 95) -> pd.DataFrame:
    factor_returns = _load_factor_returns_oos()

    panel_a = build_panel_a_table_9(factor_returns=factor_returns, confidence_level=beta / 100.0)
    panel_b = build_panel_b_or_c_table_9(
        factor_returns=factor_returns,
        forecast_groups=FORECAST_GROUPS_CVAR,
        panel_name="Panel B",
        panel_label="long_only",
        beta=beta,
    )
    panel_c = build_panel_b_or_c_table_9(
        factor_returns=factor_returns,
        forecast_groups=FORECAST_GROUPS_CVAR,
        panel_name="Panel C",
        panel_label="130_30",
        beta=beta,
        model_suffix="-S",
        total_average_label="Total Average - S",
        dependence_suffix="-S",
    )

    table = pd.concat([panel_a, panel_b, panel_c], ignore_index=True)
    table[TABLE9_METRIC_COLUMNS] = table[TABLE9_METRIC_COLUMNS].round(3)
    return table


def build_table_10() -> pd.DataFrame:
    return build_table_9(beta=99)


def _format_metric(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.3f}"


def _panel_text(
    panel_title: str,
    panel_df: pd.DataFrame,
    metric_columns: list[str],
    first_col_label: str = "Models",
) -> str:
    widths = {
        first_col_label: max(len(first_col_label), int(panel_df["Models"].astype(str).map(len).max())),
    }
    for column in metric_columns:
        widths[column] = len(column)

    lines = [panel_title]
    header = f"{first_col_label:<{widths[first_col_label]}}"
    for column in metric_columns:
        header += f"  {column:>{widths[column]}}"
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in panel_df.iterrows():
        line = f"{row['Models']:<{widths[first_col_label]}}"
        for column in metric_columns:
            line += f"  {_format_metric(row[column]):>{widths[column]}}"
        lines.append(line)

    return "\n".join(lines)


def _panel_image(
    ax: plt.Axes,
    panel_title: str,
    panel_df: pd.DataFrame,
    metric_columns: list[str],
    first_col_label: str = "Models",
) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.08,
        panel_title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    display_df = panel_df.copy()
    display_df = display_df.rename(columns={"Models": first_col_label})
    for column in metric_columns:
        display_df[column] = display_df[column].map(_format_metric)

    n_cols = len(display_df.columns)
    first_col_width = 0.24 if n_cols == 6 else 0.28
    other_width = (1.0 - first_col_width) / (n_cols - 1)
    col_widths = [first_col_width] + [other_width] * (n_cols - 1)

    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=display_df.columns.tolist(),
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 1.0],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)

    header_color = "#d9d9d9"
    stripe_color = "#f2f2f2"
    edge_color = "#6b6b6b"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.6)

        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
        else:
            if row % 2 == 0:
                cell.set_facecolor(stripe_color)
            if col == 0:
                cell.set_text_props(ha="left")


def _write_image(
    table_df: pd.DataFrame,
    png_name: str,
    title: str,
    panel_titles: dict[str, str],
    metric_columns: list[str],
    panel_first_col_labels: dict[str, str] | None = None,
) -> Path:
    png_path = DIR_TABLES / png_name
    panel_order = ["Panel A", "Panel B", "Panel C"]
    first_col_labels = panel_first_col_labels or {panel_key: "Models" for panel_key in panel_order}
    panel_frames = [
        table_df.loc[table_df["Panel"] == panel_key, ["Models", *metric_columns]].copy()
        for panel_key in panel_order
    ]

    height_ratios = [len(frame) + 2 for frame in panel_frames]
    figure_height = 0.55 * sum(height_ratios) + 1.8

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(15, figure_height),
        dpi=300,
        gridspec_kw={"height_ratios": height_ratios},
    )

    fig.suptitle(
        title,
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )

    for ax, panel_key, panel_df in zip(axes, panel_order, panel_frames):
        _panel_image(
            ax=ax,
            panel_title=panel_titles[panel_key],
            panel_df=panel_df,
            metric_columns=metric_columns,
            first_col_label=first_col_labels[panel_key],
        )

    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=2.0)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _write_image_table_8(table_8: pd.DataFrame) -> Path:
    panel_titles = {
        "Panel A": "Panel A: Factors and 1/N portfolio",
        "Panel B": "Panel B: Mean-Variance optimization without short-selling",
        "Panel C": "Panel C: Mean-Variance optimization with short-selling (130/30 portfolios)",
    }
    return _write_image(
        table_df=table_8,
        png_name="table_8.png",
        title="TABLE 8  Performances of different trading strategies (mean-variance)",
        panel_titles=panel_titles,
        metric_columns=METRIC_COLUMNS,
        panel_first_col_labels={"Panel A": "Models", "Panel B": "Models", "Panel C": "Models"},
    )


def _write_image_table_9(table_9: pd.DataFrame, beta: int = 95) -> Path:
    panel_titles = {
        "Panel A": "Panel A: Factors and 1/N portfolio",
        "Panel B": "Panel B: Mean-CVaR optimization without short-selling",
        "Panel C": "Panel C: Mean-CVaR optimization with short-selling (130/30 portfolios)",
    }
    return _write_image(
        table_df=table_9,
        png_name="table_9.png",
        title=f"TABLE 9  Performances of different trading strategies (mean-{beta}% CVaR)",
        panel_titles=panel_titles,
        metric_columns=TABLE9_METRIC_COLUMNS,
        panel_first_col_labels={"Panel A": "Factors", "Panel B": "Models", "Panel C": "Models"},
    )


def _write_image_table_10(table_10: pd.DataFrame) -> Path:
    panel_titles = {
        "Panel A": "Panel A: Factors and 1/N portfolio",
        "Panel B": "Panel B: Mean-CVaR optimization without short-selling",
        "Panel C": "Panel C: Mean-CVaR optimization with short-selling (130/30 portfolios)",
    }
    return _write_image(
        table_df=table_10,
        png_name="table_10.png",
        title="TABLE 10  Performances of different trading strategies (mean-99% CVaR)",
        panel_titles=panel_titles,
        metric_columns=TABLE9_METRIC_COLUMNS,
        panel_first_col_labels={"Panel A": "Factors", "Panel B": "Models", "Panel C": "Models"},
    )


def write_outputs(table_8: pd.DataFrame) -> None:
    DIR_TABLES.mkdir(parents=True, exist_ok=True)

    csv_path = DIR_TABLES / "table_8.csv"
    txt_path = DIR_TABLES / "table_8.txt"
    png_path = _write_image_table_8(table_8)

    table_8.to_csv(csv_path, index=False)

    panel_titles = {
        "Panel A": "Panel A: Factors and 1/N portfolio",
        "Panel B": "Panel B: Mean-Variance optimization without short-selling",
        "Panel C": "Panel C: Mean-Variance optimization with short-selling (130/30 portfolios)",
    }
    text_blocks = ["TABLE 8 - Performances of different trading strategies (mean-variance)"]
    for panel_key in ["Panel A", "Panel B", "Panel C"]:
        panel_df = table_8.loc[table_8["Panel"] == panel_key, ["Models", *METRIC_COLUMNS]]
        text_blocks.append(_panel_text(panel_titles[panel_key], panel_df, METRIC_COLUMNS, first_col_label="Models"))

    txt_path.write_text("\n\n".join(text_blocks) + "\n", encoding="utf-8")
    print("\n\n".join(text_blocks))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {png_path}")


def write_outputs_table_9(table_9: pd.DataFrame, beta: int = 95) -> None:
    DIR_TABLES.mkdir(parents=True, exist_ok=True)

    csv_path = DIR_TABLES / "table_9.csv"
    txt_path = DIR_TABLES / "table_9.txt"
    png_path = _write_image_table_9(table_9, beta=beta)

    table_9.to_csv(csv_path, index=False)

    panel_titles = {
        "Panel A": "Panel A: Factors and 1/N portfolio",
        "Panel B": "Panel B: Mean-CVaR optimization without short-selling",
        "Panel C": "Panel C: Mean-CVaR optimization with short-selling (130/30 portfolios)",
    }
    text_blocks = [f"TABLE 9 - Performances of different trading strategies (mean-{beta}% CVaR)"]
    for panel_key in ["Panel A", "Panel B", "Panel C"]:
        panel_df = table_9.loc[table_9["Panel"] == panel_key, ["Models", *TABLE9_METRIC_COLUMNS]]
        first_col_label = "Factors" if panel_key == "Panel A" else "Models"
        text_blocks.append(
            _panel_text(
                panel_titles[panel_key],
                panel_df,
                TABLE9_METRIC_COLUMNS,
                first_col_label=first_col_label,
            )
        )

    txt_path.write_text("\n\n".join(text_blocks) + "\n", encoding="utf-8")
    print("\n\n".join(text_blocks))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {png_path}")


def write_outputs_table_10(table_10: pd.DataFrame) -> None:
    DIR_TABLES.mkdir(parents=True, exist_ok=True)

    csv_path = DIR_TABLES / "table_10.csv"
    txt_path = DIR_TABLES / "table_10.txt"
    png_path = _write_image_table_10(table_10)

    table_10.to_csv(csv_path, index=False)

    panel_titles = {
        "Panel A": "Panel A: Factors and 1/N portfolio",
        "Panel B": "Panel B: Mean-CVaR optimization without short-selling",
        "Panel C": "Panel C: Mean-CVaR optimization with short-selling (130/30 portfolios)",
    }
    text_blocks = ["TABLE 10 - Performances of different trading strategies (mean-99% CVaR)"]
    for panel_key in ["Panel A", "Panel B", "Panel C"]:
        panel_df = table_10.loc[table_10["Panel"] == panel_key, ["Models", *TABLE9_METRIC_COLUMNS]]
        first_col_label = "Factors" if panel_key == "Panel A" else "Models"
        text_blocks.append(
            _panel_text(
                panel_titles[panel_key],
                panel_df,
                TABLE9_METRIC_COLUMNS,
                first_col_label=first_col_label,
            )
        )

    txt_path.write_text("\n\n".join(text_blocks) + "\n", encoding="utf-8")
    print("\n\n".join(text_blocks))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {png_path}")


def main() -> None:
    table_8 = build_table_8()
    write_outputs(table_8)
    print()
    table_9 = build_table_9(beta=95)
    write_outputs_table_9(table_9, beta=95)
    print()
    table_10 = build_table_10()
    write_outputs_table_10(table_10)


if __name__ == "__main__":
    main()
