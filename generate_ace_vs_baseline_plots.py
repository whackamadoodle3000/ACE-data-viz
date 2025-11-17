#!/usr/bin/env python3

"""
Generate BFCL comparison plots across multiple ACE configurations.

The script aligns the CSV summaries produced for:
  • Baseline (`score_baseline_all`)
  • ACE w/o Claude (`score_playbook_all`)
  • ACE w/ Claude (`score_ace_claude_run`)
  • Dynamic ACE w/ Claude (`results_sai/score_ace_claude_run`)
  • Dynamic ACE w/o Claude (`results_sai/score_ace_eval_run`)

Web-search related columns are ignored so the visuals focus on execution,
reasoning, and hallucination signals.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "plots"

RUN_SPECS: Sequence[Tuple[str, Path]] = [
    ("Baseline", BASE_DIR / "score_baseline_all"),
    ("ACE w/o Claude", BASE_DIR / "score_playbook_all"),
    ("ACE w/ Claude", BASE_DIR / "score_ace_claude_run"),
    ("Dynamic ACE w/ Claude", BASE_DIR / "results_sai" / "score_ace_claude_run"),
    ("Dynamic ACE w/o Claude", BASE_DIR / "results_sai" / "score_ace_eval_run"),
]


def normalize_value(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return np.nan

    stripped = value.strip()
    if stripped == "" or stripped.upper() == "N/A":
        return np.nan
    if stripped.endswith("%"):
        stripped = stripped[:-1]
    try:
        return float(stripped)
    except ValueError:
        return np.nan


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


def build_palette(run_names: Sequence[str]) -> Dict[str, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    colors = {}
    for idx, name in enumerate(run_names):
        colors[name] = cmap(idx % cmap.N)
    return colors


def configure_percent_axis(ax, data: Sequence[float]) -> None:
    finite_values = [val for val in data if np.isfinite(val)]
    if not finite_values:
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score (%)")
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


def build_bar_labels(
    values: Sequence[float],
    baseline_values: Sequence[float],
    unit: str,
) -> List[str]:
    labels: List[str] = []
    for idx, (value, baseline_val) in enumerate(zip(values, baseline_values)):
        if not np.isfinite(value):
            labels.append("N/A")
            continue
        text = format_value(value, unit)
        if idx > 0 and np.isfinite(baseline_val):
            delta = value - baseline_val
            if not math.isclose(delta, 0.0, abs_tol=1e-9):
                suffix = " pp" if unit == "percent" else ""
                text = f"{text}\n({delta:+.2f}{suffix})"
        labels.append(text)
    return labels


def annotate_bars(ax, bars: Iterable[plt.Rectangle], labels: Sequence[str]) -> None:
    if not labels:
        return
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for bar, label in zip(bars, labels):
        if not label:
            continue
        height = bar.get_height()
        if not np.isfinite(height):
            height = 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02 * ymax,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def load_run_data() -> OrderedDict[str, Dict[str, pd.Series]]:
    run_data: OrderedDict[str, Dict[str, pd.Series]] = OrderedDict()
    for run_name, run_path in RUN_SPECS:
        required_files = {
            "overall": run_path / "data_overall.csv",
            "non_live": run_path / "data_non_live.csv",
            "live": run_path / "data_live.csv",
            "multi": run_path / "data_multi_turn.csv",
        }
        for file_path in required_files.values():
            if not file_path.exists():
                raise FileNotFoundError(f"Missing expected file: {file_path}")

        overall = read_single_row_csv(required_files["overall"])
        overall["Composite Acc (No Web)"] = compute_composite_without_web(overall)
        run_data[run_name] = {
            "overall": overall,
            "non_live": read_single_row_csv(required_files["non_live"]),
            "live": read_single_row_csv(required_files["live"]),
            "multi": read_single_row_csv(required_files["multi"]),
        }

    return run_data


def plot_metric_panel(
    ax: plt.Axes,
    run_data: Mapping[str, Dict[str, pd.Series]],
    dataset_key: str,
    column: str,
    title: str,
    unit: str,
    colors: Mapping[str, Tuple[float, float, float, float]],
) -> None:
    run_names = list(run_data.keys())
    values = [run_data[name][dataset_key].get(column, np.nan) for name in run_names]
    heights = [val if np.isfinite(val) else 0.0 for val in values]

    x = np.arange(len(run_names))
    bars = ax.bar(
        x,
        heights,
        color=[colors[name] for name in run_names],
        width=0.6,
    )

    if unit == "percent":
        configure_percent_axis(ax, values)
    else:
        configure_value_axis(ax, values, title)

    baseline_values = [values[0]] * len(values)
    labels = build_bar_labels(values, baseline_values, unit)
    annotate_bars(ax, bars, labels)

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=20, ha="right")
    ax.set_title(title, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def render_metric_groups(
    filename: str,
    title: str,
    grouped_metrics: Sequence[Tuple[str, str, Sequence[Tuple[str, str]]]],
    run_data: Mapping[str, Dict[str, pd.Series]],
    colors: Mapping[str, Tuple[float, float, float, float]],
    unit: str = "percent",
    rotation: int = 25,
    figsize: Tuple[float, float] = (16, 6),
    ylabel: str | None = None,
    annotate: bool = True,
    show_group_labels: bool = True,
) -> None:
    run_names = list(run_data.keys())
    if not run_names:
        return

    prepared_groups: List[Tuple[str, List[Tuple[str, List[float]]]]] = []
    for group_label, dataset_key, metrics in grouped_metrics:
        group_metrics: List[Tuple[str, List[float]]] = []
        for metric_label, column in metrics:
            metric_values = [
                run_data[name][dataset_key].get(column, np.nan) for name in run_names
            ]
            if any(np.isfinite(val) for val in metric_values):
                group_metrics.append((metric_label, metric_values))
        if group_metrics:
            prepared_groups.append((group_label, group_metrics))

    if not prepared_groups:
        return

    positions: List[float] = []
    labels: List[str] = []
    values_by_run: Dict[str, List[float]] = {name: [] for name in run_names}
    group_centers: List[Tuple[str, float]] = []
    separators: List[float] = []

    current = 0.0
    for idx, (group_label, metrics) in enumerate(prepared_groups):
        start = current
        for metric_label, metric_values in metrics:
            positions.append(current)
            labels.append(metric_label)
            for name, value in zip(run_names, metric_values):
                values_by_run[name].append(value)
            current += 1.0
        center = (start + (current - 1.0)) / 2.0
        group_centers.append((group_label, center))
        if idx < len(prepared_groups) - 1:
            separators.append(current - 0.5)
        current += 0.6

    x = np.array(positions)
    num_runs = len(run_names)
    width = min(0.18, 0.8 / max(num_runs, 1))

    fig, ax = plt.subplots(figsize=figsize)
    all_values: List[float] = []
    baseline_values = values_by_run[run_names[0]]
    bar_records: List[Tuple[Iterable[plt.Rectangle], List[float]]] = []

    for idx, name in enumerate(run_names):
        values = values_by_run[name]
        heights = [val if np.isfinite(val) else 0.0 for val in values]
        offsets = (idx - (num_runs - 1) / 2.0) * width
        bars = ax.bar(
            x + offsets,
            heights,
            width,
            color=colors[name],
            label=name,
        )
        bar_records.append((bars, values))
        all_values.extend([val for val in values if np.isfinite(val)])

    if unit == "percent":
        configure_percent_axis(ax, all_values)
    else:
        configure_value_axis(ax, all_values, ylabel or title)

    if annotate:
        for idx, (bars, values) in enumerate(bar_records):
            labels_for_bars = build_bar_labels(values, baseline_values, unit)
            annotate_bars(ax, bars, labels_for_bars)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    if show_group_labels:
        for sep in separators:
            ax.axvline(sep, color="gray", linestyle=":", linewidth=1, alpha=0.6)

        for group_label, center in group_centers:
            ax.text(
                center,
                -0.15,
                group_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=11,
                fontweight="semibold",
            )

    bottom_margin = 0.28 if show_group_labels else 0.18
    fig.subplots_adjust(bottom=bottom_margin)
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def plot_summary_panels(
    run_data: Mapping[str, Dict[str, pd.Series]],
    colors: Mapping[str, Tuple[float, float, float, float]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panels = [
        ("overall", "Composite Acc (No Web)", "Composite Accuracy (No Web)", "percent"),
        ("overall", "Non-Live AST Acc", "Non-Live Accuracy", "percent"),
        ("overall", "Live Acc", "Live Accuracy", "percent"),
        ("overall", "Multi Turn Acc", "Multi-Turn Accuracy", "percent"),
    ]
    for ax, (dataset, column, title, unit) in zip(axes.flatten(), panels):
        plot_metric_panel(ax, run_data, dataset, column, title, unit, colors)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(run_data), fontsize=10)
    fig.suptitle("BFCL Macro Performance Comparison", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "summary_performance.png", dpi=300)
    plt.close(fig)


def plot_operational_metrics(
    run_data: Mapping[str, Dict[str, pd.Series]],
    colors: Mapping[str, Tuple[float, float, float, float]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        ("overall", "Total Cost ($)", "Total Cost ($)", "value"),
        ("overall", "Latency Mean (s)", "Latency Mean (s)", "value"),
        ("overall", "Latency 95th Percentile (s)", "Latency P95 (s)", "value"),
    ]
    for ax, (dataset, column, title, unit) in zip(axes, panels):
        plot_metric_panel(ax, run_data, dataset, column, title, unit, colors)
        ax.set_ylabel(title)
    fig.suptitle("Operational Metrics (Lower Is Better)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "operational_metrics.png", dpi=300)
    plt.close(fig)


def plot_non_live_breakdown(
    run_data: Mapping[str, Dict[str, pd.Series]],
    colors: Mapping[str, Tuple[float, float, float, float]],
) -> None:
    grouped_metrics = [
        (
            "Non-Live",
            "non_live",
            [
                ("Overall Accuracy", "Non-Live Overall Acc"),
                ("AST Summary", "AST Summary"),
                ("Simple AST", "Simple AST"),
                ("Python Simple AST", "Python Simple AST"),
                ("Java Simple AST", "Java Simple AST"),
                ("JavaScript Simple AST", "JavaScript Simple AST"),
                ("Multiple AST", "Multiple AST"),
                ("Parallel AST", "Parallel AST"),
                ("Parallel Multiple AST", "Parallel Multiple AST"),
                ("Irrelevance Detection", "Irrelevance Detection"),
            ],
        ),
    ]
    render_metric_groups(
        "non_live_breakdown.png",
        "Non-Live Task Breakdown (Accuracy %)",
        grouped_metrics,
        run_data,
        colors,
    )


def plot_live_breakdown(
    run_data: Mapping[str, Dict[str, pd.Series]],
    colors: Mapping[str, Tuple[float, float, float, float]],
) -> None:
    grouped_metrics = [
        (
            "Live",
            "live",
            [
                ("Overall Accuracy", "Live Overall Acc"),
                ("AST Summary", "AST Summary"),
                ("Python Simple AST", "Python Simple AST"),
                ("Python Multiple AST", "Python Multiple AST"),
                ("Python Parallel AST", "Python Parallel AST"),
                ("Python Parallel Multiple AST", "Python Parallel Multiple AST"),
                ("Irrelevance Detection", "Irrelevance Detection"),
                ("Relevance Detection", "Relevance Detection"),
            ],
        ),
    ]
    render_metric_groups(
        "live_breakdown.png",
        "Live Task Breakdown (Accuracy %)",
        grouped_metrics,
        run_data,
        colors,
    )


def plot_multi_turn_breakdown(
    run_data: Mapping[str, Dict[str, pd.Series]],
    colors: Mapping[str, Tuple[float, float, float, float]],
) -> None:
    grouped_metrics = [
        (
            "Multi-Turn",
            "multi",
            [
                ("Overall Accuracy", "Multi Turn Overall Acc"),
                ("Base Tasks", "Base"),
                ("Missing Function", "Miss Func"),
                ("Missing Param", "Miss Param"),
                ("Long Context", "Long Context"),
            ],
        ),
    ]
    render_metric_groups(
        "multi_turn_breakdown.png",
        "Multi-Turn Task Breakdown (Accuracy %)",
        grouped_metrics,
        run_data,
        colors,
        figsize=(13, 5),
    )


def plot_detection_overview(
    run_data: Mapping[str, Dict[str, pd.Series]],
    colors: Mapping[str, Tuple[float, float, float, float]],
) -> None:
    grouped_metrics = [
        (
            "Overall",
            "overall",
            [
                ("Relevance Detection", "Relevance Detection"),
                ("Irrelevance Detection", "Irrelevance Detection"),
            ],
        ),
        (
            "Live",
            "live",
            [
                ("Relevance Detection", "Relevance Detection"),
                ("Irrelevance Detection", "Irrelevance Detection"),
            ],
        ),
        (
            "Non-Live",
            "non_live",
            [
                ("Irrelevance Detection", "Irrelevance Detection"),
            ],
        ),
    ]
    render_metric_groups(
        "detection_overview.png",
        "Relevance vs Irrelevance Detection (Accuracy %)",
        grouped_metrics,
        run_data,
        colors,
    )


def plot_composite_breakdown(
    run_data: Mapping[str, Dict[str, pd.Series]],
    colors: Mapping[str, Tuple[float, float, float, float]],
) -> None:
    grouped_metrics = [
        (
            "Composite",
            "overall",
            [
                ("Composite Accuracy", "Composite Acc (No Web)"),
            ],
        ),
        (
            "Non-Live",
            "non_live",
            [
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
            ],
        ),
        (
            "Live",
            "live",
            [
                ("Overall", "Live Overall Acc"),
                ("AST Summary", "AST Summary"),
                ("Python Simple AST", "Python Simple AST"),
                ("Python Multiple AST", "Python Multiple AST"),
                ("Python Parallel AST", "Python Parallel AST"),
                ("Python Parallel Multiple AST", "Python Parallel Multiple AST"),
                ("Irrelevance Detection", "Irrelevance Detection"),
                ("Relevance Detection", "Relevance Detection"),
            ],
        ),
        (
            "Multi-Turn",
            "multi",
            [
                ("Overall", "Multi Turn Overall Acc"),
                ("Base Tasks", "Base"),
                ("Missing Function", "Miss Func"),
                ("Missing Param", "Miss Param"),
                ("Long Context", "Long Context"),
            ],
        ),
    ]
    render_metric_groups(
        "bfcl_composite_breakdown.png",
        "BFCL Composite Breakdown (Accuracy %)",
        grouped_metrics,
        run_data,
        colors,
        rotation=35,
        figsize=(20, 8),
    )


def plot_grouped_composite_view(
    run_data: Mapping[str, Dict[str, pd.Series]],
    colors: Mapping[str, Tuple[float, float, float, float]],
) -> None:
    grouped_metrics = [
        ("Composite", "overall", [("Composite", "Composite Acc (No Web)")]),
        ("Non-Live Single-Turn", "non_live", [("Non-Live Single-Turn", "Non-Live Overall Acc")]),
        ("Live Single-Turn", "live", [("Live Single-Turn", "Live Overall Acc")]),
        ("Multi-Turn", "multi", [("Multi-Turn", "Multi Turn Overall Acc")]),
        ("Relevance", "overall", [("Relevance", "Relevance Detection")]),
        ("Irrelevance", "overall", [("Irrelevance", "Irrelevance Detection")]),
    ]
    render_metric_groups(
        "bfcl_grouped_overview.png",
        "BFCL Core Groupings (Accuracy %)",
        grouped_metrics,
        run_data,
        colors,
        figsize=(14, 6),
        annotate=False,
        show_group_labels=False,
        rotation=15,
    )


def build_summary_table(run_data: Mapping[str, Dict[str, pd.Series]]) -> None:
    run_names = list(run_data.keys())
    metrics = [
        ("Composite Accuracy (No Web) (%)", "overall", "Composite Acc (No Web)", "percent"),
        ("Non-Live Accuracy (%)", "overall", "Non-Live AST Acc", "percent"),
        ("Live Accuracy (%)", "overall", "Live Acc", "percent"),
        ("Multi-Turn Accuracy (%)", "overall", "Multi Turn Acc", "percent"),
        ("Total Cost ($)", "overall", "Total Cost ($)", "value"),
        ("Latency Mean (s)", "overall", "Latency Mean (s)", "value"),
        ("Latency P95 (s)", "overall", "Latency 95th Percentile (s)", "value"),
        ("Non-Live Irrelevance Detection (%)", "non_live", "Irrelevance Detection", "percent"),
        ("Live Relevance Detection (%)", "live", "Relevance Detection", "percent"),
        ("Multi-Turn Base Accuracy (%)", "multi", "Base", "percent"),
    ]

    headers = ["Metric"] + run_names
    table_rows: List[List[str]] = []

    for metric_label, dataset_key, column, unit in metrics:
        row = [metric_label]
        has_value = False
        for run_name in run_names:
            value = run_data[run_name][dataset_key].get(column, np.nan)
            formatted = format_value(value, unit)
            if formatted != "N/A":
                has_value = True
            row.append(formatted)
        if has_value:
            table_rows.append(row)

    if not table_rows:
        return

    fig, ax = plt.subplots(figsize=(3 + 2.2 * len(run_names), 4))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    fig.suptitle("Key Metrics Summary Table", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary_table.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_output_dir()
    run_data = load_run_data()
    colors = build_palette(list(run_data.keys()))

    plot_summary_panels(run_data, colors)
    plot_operational_metrics(run_data, colors)
    plot_non_live_breakdown(run_data, colors)
    plot_live_breakdown(run_data, colors)
    plot_multi_turn_breakdown(run_data, colors)
    plot_detection_overview(run_data, colors)
    plot_composite_breakdown(run_data, colors)
    plot_grouped_composite_view(run_data, colors)
    build_summary_table(run_data)

    print(f"Plots saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

