#!/usr/bin/env python3

"""
Generate detailed comparison plots between the baseline and ACE evaluation runs.

This script reads the CSV summaries stored under `score_baseline_all` and
`score_playbook_all`, cleans and aligns the numeric metrics, and produces a
series of Matplotlib visualizations that highlight how the ACE configuration
performs relative to the baseline on the same dataset.

Web-search related metrics are intentionally excluded to keep the focus on the
core task-oriented capabilities.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
BASELINE_DIR = BASE_DIR / "score_baseline_all"
ACE_DIR = BASE_DIR / "score_playbook_all"
OUTPUT_DIR = BASE_DIR / "plots"

BASELINE_COLOR = "#1f77b4"
ACE_COLOR = "#ff7f0e"


def normalize_value(value) -> float | str:
    """Convert percentage strings and 'N/A' markers into numeric floats."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped == "" or stripped.upper() == "N/A":
        return np.nan
    if stripped.endswith("%"):
        stripped = stripped[:-1]
        if stripped == "":
            return np.nan
        try:
            return float(stripped)
        except ValueError:
            return np.nan
    try:
        return float(stripped)
    except ValueError:
        return stripped


def read_single_row_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No data in {path}")
    normalized = {key: normalize_value(value) for key, value in df.iloc[0].items()}
    return pd.Series(normalized)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_value(value: float, unit: str) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if unit == "percent":
        return f"{value:.2f}%"
    return f"{value:.2f}"


def delta_label(delta: float, unit: str) -> str:
    if delta is None or pd.isna(delta):
        return "ACE − Baseline: N/A"
    suffix = " pp" if unit == "percent" else ""
    return f"ACE − Baseline: {delta:+.2f}{suffix}"


def configure_percent_axis(ax, data: Sequence[float]) -> None:
    finite_values = [val for val in data if np.isfinite(val)]
    if not finite_values:
        ax.set_ylim(0, 1)
        return
    ymax = max(finite_values)
    upper = max(100.0, ymax * 1.15)
    ax.set_ylim(0, upper)
    ax.set_ylabel("Score (%)")


def configure_value_axis(ax, data: Sequence[float], label: str) -> None:
    finite_values = [val for val in data if np.isfinite(val)]
    if not finite_values:
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        return
    ymax = max(finite_values)
    upper = ymax * 1.2 if ymax > 0 else 1
    ax.set_ylim(0, upper)
    ax.set_ylabel(label)


def annotate_bars(ax, bars: Iterable[plt.Rectangle], unit: str) -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            label = "N/A"
            height = 0
        else:
            label = format_value(height, unit)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.02 * ax.get_ylim()[1]),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def compute_composite_without_web(overall: pd.Series) -> float:
    components = [
        overall.get("Non-Live AST Acc", np.nan),
        overall.get("Live Acc", np.nan),
        overall.get("Multi Turn Acc", np.nan),
    ]
    finite = [val for val in components if np.isfinite(val)]
    if not finite:
        return np.nan
    return float(np.mean(finite))


def plot_single_metric(
    ax: plt.Axes,
    label: str,
    column: str,
    unit: str,
    baseline: pd.Series,
    ace: pd.Series,
) -> None:
    baseline_val = baseline[column]
    ace_val = ace[column]
    values = [baseline_val, ace_val]

    if unit == "percent":
        configure_percent_axis(ax, values)
    else:
        configure_value_axis(ax, values, label)

    bars = ax.bar(
        [0, 1],
        values,
        color=[BASELINE_COLOR, ACE_COLOR],
        width=0.6,
    )
    ax.set_xticks([0, 1], ["Baseline", "ACE"])
    ax.set_title(label, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    annotate_bars(ax, bars, unit)

    if np.isfinite(baseline_val) and np.isfinite(ace_val):
        delta = ace_val - baseline_val
        ax.text(
            0.5,
            ax.get_ylim()[1] * 0.92,
            delta_label(delta, unit),
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
        )


def plot_grouped_metrics(
    ax: plt.Axes,
    metrics: Sequence[Tuple[str, str, str]],
    baseline: pd.Series,
    ace: pd.Series,
    title: str,
) -> None:
    labels: List[str] = []
    baseline_vals: List[float] = []
    ace_vals: List[float] = []

    for display, column, unit in metrics:
        base_val = baseline.get(column, np.nan)
        ace_val = ace.get(column, np.nan)
        if (not np.isfinite(base_val)) and (not np.isfinite(ace_val)):
            continue
        labels.append(display)
        baseline_vals.append(base_val)
        ace_vals.append(ace_val)

    if not labels:
        ax.text(0.5, 0.5, "No comparable metrics available.", ha="center", va="center")
        ax.set_axis_off()
        return

    unit = metrics[0][2]
    x = np.arange(len(labels))
    width = 0.35

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline",
        color=BASELINE_COLOR,
    )
    bars_ace = ax.bar(
        x + width / 2,
        ace_vals,
        width,
        label="ACE",
        color=ACE_COLOR,
    )

    if unit == "percent":
        configure_percent_axis(ax, baseline_vals + ace_vals)
    else:
        configure_value_axis(ax, baseline_vals + ace_vals, title)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    annotate_bars(ax, bars_baseline, unit)
    annotate_bars(ax, bars_ace, unit)

    ylim_upper = ax.get_ylim()[1]
    for idx, (base_val, ace_val) in enumerate(zip(baseline_vals, ace_vals)):
        if np.isfinite(base_val) and np.isfinite(ace_val):
            delta = ace_val - base_val
            ax.text(
                x[idx],
                ylim_upper * 0.9,
                delta_label(delta, unit),
                ha="center",
                va="top",
                fontsize=9,
                rotation=90,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )


def plot_summary_panels(baseline: pd.Series, ace: pd.Series) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("Composite Accuracy (No Web Search)", "Composite Acc (No Web)", "percent"),
        ("Non-Live Accuracy", "Non-Live AST Acc", "percent"),
        ("Live Accuracy", "Live Acc", "percent"),
        ("Multi-Turn Accuracy", "Multi Turn Acc", "percent"),
    ]
    for ax, args in zip(axes.flatten(), panels):
        plot_single_metric(ax, *args, baseline, ace)

    fig.suptitle("Macro Performance Comparison (Baseline vs ACE)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_DIR / "summary_performance.png", dpi=300)
    plt.close(fig)


def plot_latency_and_cost(baseline: pd.Series, ace: pd.Series) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        ("Total Cost ($)", "Total Cost ($)", "dollars"),
        ("Latency Mean (s)", "Latency Mean (s)", "Seconds"),
        ("Latency 95th Percentile (s)", "Latency 95th Percentile (s)", "Seconds"),
    ]
    for ax, (label, column, unit_label) in zip(axes, panels):
        plot_single_metric(ax, label, column, "value", baseline, ace)
        ax.set_ylabel(unit_label)
    fig.suptitle("Operational Metrics (Baseline vs ACE)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "operational_metrics.png", dpi=300)
    plt.close(fig)


def plot_non_live_breakdown(baseline: pd.Series, ace: pd.Series) -> None:
    metrics = [
        ("Overall Accuracy", "Non-Live Overall Acc", "percent"),
        ("AST Summary", "AST Summary", "percent"),
        ("Simple AST", "Simple AST", "percent"),
        ("Python Simple AST", "Python Simple AST", "percent"),
        ("Java Simple AST", "Java Simple AST", "percent"),
        ("JavaScript Simple AST", "JavaScript Simple AST", "percent"),
        ("Multiple AST", "Multiple AST", "percent"),
        ("Parallel AST", "Parallel AST", "percent"),
        ("Parallel Multiple AST", "Parallel Multiple AST", "percent"),
        ("Irrelevance Detection", "Irrelevance Detection", "percent"),
    ]
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_grouped_metrics(
        ax,
        metrics,
        baseline,
        ace,
        "Non-Live Task Breakdown (Accuracy %)",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "non_live_breakdown.png", dpi=300)
    plt.close(fig)


def plot_live_breakdown(baseline: pd.Series, ace: pd.Series) -> None:
    metrics = [
        ("Overall Accuracy", "Live Overall Acc", "percent"),
        ("AST Summary", "AST Summary", "percent"),
        ("Python Simple AST", "Python Simple AST", "percent"),
        ("Python Multiple AST", "Python Multiple AST", "percent"),
        ("Python Parallel AST", "Python Parallel AST", "percent"),
        ("Python Parallel Multiple AST", "Python Parallel Multiple AST", "percent"),
        ("Irrelevance Detection", "Irrelevance Detection", "percent"),
        ("Relevance Detection", "Relevance Detection", "percent"),
    ]
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_grouped_metrics(
        ax,
        metrics,
        baseline,
        ace,
        "Live Task Breakdown (Accuracy %)",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "live_breakdown.png", dpi=300)
    plt.close(fig)


def plot_multi_turn_breakdown(baseline: pd.Series, ace: pd.Series) -> None:
    metrics = [
        ("Overall Accuracy", "Multi Turn Overall Acc", "percent"),
        ("Base Tasks", "Base", "percent"),
        ("Missing Function", "Miss Func", "percent"),
        ("Missing Param", "Miss Param", "percent"),
        ("Long Context", "Long Context", "percent"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_grouped_metrics(
        ax,
        metrics,
        baseline,
        ace,
        "Multi-Turn Task Breakdown (Accuracy %)",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "multi_turn_breakdown.png", dpi=300)
    plt.close(fig)


def plot_detection_overview(
    overall_baseline: pd.Series,
    overall_ace: pd.Series,
    live_baseline: pd.Series,
    live_ace: pd.Series,
    non_live_baseline: pd.Series,
    non_live_ace: pd.Series,
) -> None:
    metrics = [
        ("Overall Relevance Detection", overall_baseline["Relevance Detection"], overall_ace["Relevance Detection"]),
        ("Overall Irrelevance Detection", overall_baseline["Irrelevance Detection"], overall_ace["Irrelevance Detection"]),
        ("Live Relevance Detection", live_baseline["Relevance Detection"], live_ace["Relevance Detection"]),
        ("Live Irrelevance Detection", live_baseline["Irrelevance Detection"], live_ace["Irrelevance Detection"]),
        ("Non-Live Irrelevance Detection", non_live_baseline["Irrelevance Detection"], non_live_ace["Irrelevance Detection"]),
    ]

    labels = []
    baseline_vals = []
    ace_vals = []
    for label, base_val, ace_val in metrics:
        if (not np.isfinite(base_val)) and (not np.isfinite(ace_val)):
            continue
        labels.append(label)
        baseline_vals.append(base_val)
        ace_vals.append(ace_val)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars_baseline = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline",
        color=BASELINE_COLOR,
    )
    bars_ace = ax.bar(
        x + width / 2,
        ace_vals,
        width,
        label="ACE",
        color=ACE_COLOR,
    )

    configure_percent_axis(ax, baseline_vals + ace_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Relevance vs Irrelevance Detection (Accuracy %)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    annotate_bars(ax, bars_baseline, "percent")
    annotate_bars(ax, bars_ace, "percent")

    ylim_upper = ax.get_ylim()[1]
    for idx, (base_val, ace_val) in enumerate(zip(baseline_vals, ace_vals)):
        if np.isfinite(base_val) and np.isfinite(ace_val):
            delta = ace_val - base_val
            ax.text(
                x[idx],
                ylim_upper * 0.88,
                delta_label(delta, "percent"),
                ha="center",
                va="top",
                fontsize=9,
                rotation=90,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "detection_overview.png", dpi=300)
    plt.close(fig)


def build_summary_table(
    overall_baseline: pd.Series,
    overall_ace: pd.Series,
    live_baseline: pd.Series,
    live_ace: pd.Series,
    non_live_baseline: pd.Series,
    non_live_ace: pd.Series,
    multi_baseline: pd.Series,
    multi_ace: pd.Series,
) -> None:
    rows = [
        ("Composite Accuracy (No Web Search) (%)", overall_baseline["Composite Acc (No Web)"], overall_ace["Composite Acc (No Web)"]),
        ("Non-Live Accuracy (%)", overall_baseline["Non-Live AST Acc"], overall_ace["Non-Live AST Acc"]),
        ("Live Accuracy (%)", overall_baseline["Live Acc"], overall_ace["Live Acc"]),
        ("Multi-Turn Accuracy (%)", overall_baseline["Multi Turn Acc"], overall_ace["Multi Turn Acc"]),
        ("Total Cost ($)", overall_baseline["Total Cost ($)"], overall_ace["Total Cost ($)"]),
        ("Latency Mean (s)", overall_baseline["Latency Mean (s)"], overall_ace["Latency Mean (s)"]),
        ("Latency P95 (s)", overall_baseline["Latency 95th Percentile (s)"], overall_ace["Latency 95th Percentile (s)"]),
        ("Non-Live Irrelevance Detection (%)", non_live_baseline["Irrelevance Detection"], non_live_ace["Irrelevance Detection"]),
        ("Live Relevance Detection (%)", live_baseline["Relevance Detection"], live_ace["Relevance Detection"]),
        ("Multi-Turn Base (%)", multi_baseline["Base"], multi_ace["Base"]),
    ]

    table_data = []
    for metric, base_val, ace_val in rows:
        if not (np.isfinite(base_val) or np.isfinite(ace_val)):
            continue
        is_percent = metric.endswith("(%)")
        unit = "percent" if is_percent else "value"
        delta = ace_val - base_val if (np.isfinite(base_val) and np.isfinite(ace_val)) else np.nan
        table_data.append(
            [
                metric,
                format_value(base_val, "percent" if is_percent else "value"),
                format_value(ace_val, "percent" if is_percent else "value"),
                delta_label(delta, "percent" if is_percent else "value").replace("ACE − Baseline: ", "") if np.isfinite(delta) else "N/A",
            ]
        )

    if not table_data:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Baseline", "ACE", "ACE − Baseline"],
        cellLoc="center",
        colLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    fig.suptitle("Key Metrics Summary Table", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary_table.png", dpi=300)
    plt.close(fig)


def plot_composite_breakdown(
    overall_baseline: pd.Series,
    overall_ace: pd.Series,
    non_live_baseline: pd.Series,
    non_live_ace: pd.Series,
    live_baseline: pd.Series,
    live_ace: pd.Series,
    multi_baseline: pd.Series,
    multi_ace: pd.Series,
) -> None:
    def add_entry(
        entries: List[Tuple[str, str, float, float]],
        group: str,
        label: str,
        baseline_val,
        ace_val,
    ) -> None:
        if not (np.isfinite(baseline_val) or np.isfinite(ace_val)):
            return
        entries.append((group, label, float(baseline_val), float(ace_val)))

    entries: List[Tuple[str, str, float, float]] = []

    add_entry(
        entries,
        "Composite Total",
        "Composite Accuracy",
        overall_baseline["Composite Acc (No Web)"],
        overall_ace["Composite Acc (No Web)"],
    )

    # Non-Live group
    for label, column in [
        ("Overall", "Non-Live Overall Acc"),
        ("AST Summary", "AST Summary"),
        ("Simple AST", "Simple AST"),
        ("Python Simple AST", "Python Simple AST"),
        ("Java Simple AST", "Java Simple AST"),
        ("JavaScript Simple AST", "JavaScript Simple AST"),
        ("Multiple AST", "Multiple AST"),
        ("Parallel AST", "Parallel AST"),
        ("Parallel Multiple AST", "Parallel Multiple AST"),
        ("Irrelevance Detection", "Irrelevance Detection"),
    ]:
        add_entry(
            entries,
            "Non-Live",
            label,
            non_live_baseline.get(column, np.nan),
            non_live_ace.get(column, np.nan),
        )

    # Live group
    for label, column in [
        ("Overall", "Live Overall Acc"),
        ("AST Summary", "AST Summary"),
        ("Simple AST", "Python Simple AST"),
        ("Multiple AST", "Python Multiple AST"),
        ("Parallel AST", "Python Parallel AST"),
        ("Parallel Multiple AST", "Python Parallel Multiple AST"),
        ("Irrelevance Detection", "Irrelevance Detection"),
        ("Relevance Detection", "Relevance Detection"),
    ]:
        add_entry(
            entries,
            "Live",
            label,
            live_baseline.get(column, np.nan),
            live_ace.get(column, np.nan),
        )

    # Multi-Turn group
    for label, column in [
        ("Overall", "Multi Turn Overall Acc"),
        ("Base Tasks", "Base"),
        ("Missing Function", "Miss Func"),
        ("Missing Param", "Miss Param"),
        ("Long Context", "Long Context"),
    ]:
        add_entry(
            entries,
            "Multi-Turn",
            label,
            multi_baseline.get(column, np.nan),
            multi_ace.get(column, np.nan),
        )

    if not entries:
        return

    groups_order: List[str] = []
    grouped_entries: Dict[str, List[Tuple[str, str, float, float]]] = {}
    for group, label, baseline_val, ace_val in entries:
        groups_order.append(group) if group not in groups_order else None
        grouped_entries.setdefault(group, []).append((group, label, baseline_val, ace_val))

    x_positions: List[float] = []
    labels: List[str] = []
    baseline_vals: List[float] = []
    ace_vals: List[float] = []
    group_centers: List[Tuple[str, float]] = []
    separators: List[float] = []

    current = 0.0
    for idx, group in enumerate(groups_order):
        items = grouped_entries[group]
        start = current
        for _, label, baseline_val, ace_val in items:
            x_positions.append(current)
            labels.append(label)
            baseline_vals.append(baseline_val)
            ace_vals.append(ace_val)
            current += 1.0
        center = (start + (current - 1.0)) / 2.0 if items else start
        group_centers.append((group, center))
        if idx < len(groups_order) - 1:
            separators.append(current - 0.5)
        current += 0.6  # gap between groups

    x_positions_np = np.array(x_positions)
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 8))
    bars_baseline = ax.bar(
        x_positions_np - width / 2,
        baseline_vals,
        width,
        color=BASELINE_COLOR,
        label="Baseline",
    )
    bars_ace = ax.bar(
        x_positions_np + width / 2,
        ace_vals,
        width,
        color=ACE_COLOR,
        label="ACE",
    )

    configure_percent_axis(ax, baseline_vals + ace_vals)
    ax.set_xticks(x_positions_np)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("BFCL Composite Breakdown (Baseline vs ACE)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    annotate_bars(ax, bars_baseline, "percent")
    annotate_bars(ax, bars_ace, "percent")

    for sep in separators:
        ax.axvline(sep, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    for group, center in group_centers:
        ax.text(
            center,
            -0.12,
            group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
            fontweight="semibold",
        )

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(OUTPUT_DIR / "bfcl_composite_breakdown.png", dpi=300)
    plt.close(fig)


def average_metrics(series: pd.Series, columns: Sequence[str]) -> float:
    values = [normalize_value(series.get(col, np.nan)) for col in columns]
    numeric = [float(val) for val in values if np.isfinite(val)]
    if not numeric:
        return np.nan
    return float(np.mean(numeric))


def plot_grouped_composite_view(
    overall_baseline: pd.Series,
    overall_ace: pd.Series,
    non_live_baseline: pd.Series,
    non_live_ace: pd.Series,
    live_baseline: pd.Series,
    live_ace: pd.Series,
    multi_baseline: pd.Series,
    multi_ace: pd.Series,
) -> None:
    categories = [
        (
            "Composite Accuracy",
            overall_baseline,
            overall_ace,
            lambda series: series["Composite Acc (No Web)"],
        ),
        (
            "Non-Live Single-Turn",
            non_live_baseline,
            non_live_ace,
            lambda series: series["Non-Live Overall Acc"],
        ),
        (
            "Live Single-Turn",
            live_baseline,
            live_ace,
            lambda series: series["Live Overall Acc"],
        ),
        (
            "Multi-Turn",
            multi_baseline,
            multi_ace,
            lambda series: series["Multi Turn Overall Acc"],
        ),
        (
            "Relevance Detection",
            overall_baseline,
            overall_ace,
            lambda series: series["Relevance Detection"],
        ),
        (
            "Irrelevance Detection",
            overall_baseline,
            overall_ace,
            lambda series: series["Irrelevance Detection"],
        ),
    ]

    rows = []
    for label, baseline_series, ace_series, extractor in categories:
        baseline_val = normalize_value(extractor(baseline_series))
        ace_val = normalize_value(extractor(ace_series))
        if np.isfinite(baseline_val) or np.isfinite(ace_val):
            rows.append((label, float(baseline_val), float(ace_val)))

    if not rows:
        return

    labels = [label for label, _, _ in rows]
    baseline_vals = [val for _, val, _ in rows]
    ace_vals = [val for _, _, val in rows]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_baseline = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline",
        color=BASELINE_COLOR,
    )
    bars_ace = ax.bar(
        x + width / 2,
        ace_vals,
        width,
        label="ACE",
        color=ACE_COLOR,
    )

    configure_percent_axis(ax, baseline_vals + ace_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("BFCL Core Groupings (Baseline vs ACE)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    annotate_bars(ax, bars_baseline, "percent")
    annotate_bars(ax, bars_ace, "percent")

    ylim_upper = ax.get_ylim()[1]
    for idx, (base_val, ace_val) in enumerate(zip(baseline_vals, ace_vals)):
        if np.isfinite(base_val) and np.isfinite(ace_val):
            delta = ace_val - base_val
            ax.text(
                x[idx],
                ylim_upper * 0.9,
                delta_label(delta, "percent"),
                ha="center",
                va="top",
                fontsize=9,
                rotation=90,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bfcl_grouped_overview.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_output_dir()

    overall_baseline = read_single_row_csv(BASELINE_DIR / "data_overall.csv")
    overall_ace = read_single_row_csv(ACE_DIR / "data_overall.csv")
    overall_baseline["Composite Acc (No Web)"] = compute_composite_without_web(overall_baseline)
    overall_ace["Composite Acc (No Web)"] = compute_composite_without_web(overall_ace)
    non_live_baseline = read_single_row_csv(BASELINE_DIR / "data_non_live.csv")
    non_live_ace = read_single_row_csv(ACE_DIR / "data_non_live.csv")
    live_baseline = read_single_row_csv(BASELINE_DIR / "data_live.csv")
    live_ace = read_single_row_csv(ACE_DIR / "data_live.csv")
    multi_baseline = read_single_row_csv(BASELINE_DIR / "data_multi_turn.csv")
    multi_ace = read_single_row_csv(ACE_DIR / "data_multi_turn.csv")

    plot_summary_panels(overall_baseline, overall_ace)
    plot_latency_and_cost(overall_baseline, overall_ace)
    plot_non_live_breakdown(non_live_baseline, non_live_ace)
    plot_live_breakdown(live_baseline, live_ace)
    plot_multi_turn_breakdown(multi_baseline, multi_ace)
    plot_detection_overview(
        overall_baseline,
        overall_ace,
        live_baseline,
        live_ace,
        non_live_baseline,
        non_live_ace,
    )
    build_summary_table(
        overall_baseline,
        overall_ace,
        live_baseline,
        live_ace,
        non_live_baseline,
        non_live_ace,
        multi_baseline,
        multi_ace,
    )
    plot_composite_breakdown(
        overall_baseline,
        overall_ace,
        non_live_baseline,
        non_live_ace,
        live_baseline,
        live_ace,
        multi_baseline,
        multi_ace,
    )
    plot_grouped_composite_view(
        overall_baseline,
        overall_ace,
        non_live_baseline,
        non_live_ace,
        live_baseline,
        live_ace,
        multi_baseline,
        multi_ace,
    )

    print(f"Plots saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


