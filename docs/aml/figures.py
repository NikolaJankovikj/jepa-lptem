"""Data-efficiency figure of the AML report, from experiments/encoder-compare/metrics.json.

uv run python docs/aml/figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
METRICS = HERE.parent.parent / "experiments/encoder-compare/metrics.json"
MODELS = {  # name: (label, colour, marker); validated categorical slots 1-3
    "frozen": ("frozen V-JEPA 2", "#2a78d6", "o"),
    "cnn": ("CNN", "#eb6834", "s"),
    "lora": ("LoRA V-JEPA 2", "#1baf7a", "^"),
}
FRACS = {10: 160, 30: 480, 100: 1600}
PANELS = (
    ("r2_count", "count $R^2$ (higher is better)"),
    ("r2_mean_radius", "mean radius $R^2$ (higher is better)"),
    ("reading_sd", "radius reading sd, px (lower is better)"),
)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def style_axis(ax, title: str) -> None:
    """Log x over the three training-set sizes, recessive grid and spines."""
    ax.set_xscale("log")
    ax.set_xticks(list(FRACS.values()), [str(n) for n in FRACS.values()])
    ax.minorticks_off()
    ax.set_xlabel("training clips")
    ax.set_title(title, fontsize=8, color=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)


def plot_panel(ax, runs: dict, stat: str, title: str) -> None:
    """One statistic against training-set size, a line per model with 95 % CIs."""
    for model, (label, colour, marker) in MODELS.items():
        cells = [runs[f"{model}-{f}"][stat] for f in FRACS]
        y = [c["value"] for c in cells]
        err = [
            [v - c["ci"][0] for v, c in zip(y, cells, strict=True)],
            [c["ci"][1] - v for v, c in zip(y, cells, strict=True)],
        ]
        ax.errorbar(
            list(FRACS.values()),
            y,
            yerr=err,
            color=colour,
            marker=marker,
            markersize=5,
            linewidth=2,
            capsize=2,
            label=label,
        )
    style_axis(ax, title)


def main() -> None:
    runs = json.loads(METRICS.read_text())["runs"]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), layout="constrained")
    for ax, (stat, title) in zip(axes, PANELS, strict=True):
        plot_panel(ax, runs, stat, title)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncols=3,
        frameon=False,
    )
    fig.savefig(HERE / "fig_data_efficiency.pdf")


if __name__ == "__main__":
    main()
