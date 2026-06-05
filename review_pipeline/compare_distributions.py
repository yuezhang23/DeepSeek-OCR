import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

OUT_DIR = Path(__file__).parent / "out"
SOURCES = [
    # ColorBrewer Set1 — high categorical contrast.
    ("deepReview", OUT_DIR / "alignment_results_review_4_standard.json", "#e41a1c"),  # red
    ("deepseek",   OUT_DIR / "ai_results.json",                          "#377eb8"),  # blue
    ("human",      OUT_DIR / "human_results_296.json",                   "#4daf4a"),  # green
]
FIG_PATH = OUT_DIR / "distribution_comparison.png"

DIMENSIONS = [
    "originality",
    "importance_of_research_question",
    "claims_well_supported",
    "soundness_of_experiments",
    "clarity_of_writing",
    "value_to_research_community",
    "contextualization_relative_to_prior_work",
]

SHORT_LABELS = [
    "originality", "importance", "claims\nsupported", "soundness\nof exp",
    "clarity", "value to\ncommunity", "context\nvs prior",
]


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def collect(data: dict, keys, dim: str):
    vals = []
    for k in keys:
        entry = data.get(k)
        if not isinstance(entry, dict):
            continue
        v = entry.get(dim)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def _kde_polygon(vals, y_grid, x_center, half_width):
    """Equal-area violin half-widths: KDE area is 1, multiplied by half_width."""
    if len(vals) < 2 or np.std(vals) < 1e-6:
        # Degenerate (all same value). Render as a thin rectangle around the value.
        center = float(np.mean(vals))
        ys = np.array([center - 0.1, center + 0.1])
        widths = np.array([half_width * 0.5, half_width * 0.5])
        return x_center - widths, x_center + widths, ys
    kde = gaussian_kde(vals, bw_method=0.4)
    densities = kde(y_grid)
    widths = densities * half_width  # area under widths(y) == half_width (since ∫kde=1)
    return x_center - widths, x_center + widths, y_grid


def draw_violin(ax, dim: str, datasets, shared) -> None:
    n_sources = len(datasets)
    # Each source gets its own slot — no overlap.
    positions = np.arange(1, n_sources + 1, dtype=float)
    # Equal area for every violin: widths(y) = kde(y) * half_width, ∫kde = 1.
    # Typical max-kde on score data is ~0.4 → max half-width ≈ half_width * 0.4.
    # 1.05 → ~0.42 visible half-width, fits in slot of 1.0 with a small gap.
    half_width = 1.05

    per_source_vals = [collect(d, shared, dim) for _, d, _ in datasets]
    y_grid = np.linspace(0.5, 10.5, 200)

    for x, vals, (_, _, color) in zip(positions, per_source_vals, datasets):
        if not vals:
            continue
        left, right, ys = _kde_polygon(vals, y_grid, x, half_width)
        ax.fill_betweenx(ys, left, right, facecolor=color, edgecolor="black",
                         linewidth=0.6, alpha=0.75)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        mean = float(np.mean(vals))
        ax.vlines(x, q1, q3, color="black", linewidth=3, zorder=3)
        ax.scatter(x, med, marker="o", color="white",
                   edgecolor="black", s=28, zorder=4)
        ax.scatter(x, mean, marker="D", color=color,
                   edgecolor="black", s=32, zorder=5)

    ax.set_title(dim.replace("_", " "), fontsize=14)
    ax.set_xticks([])
    ax.set_xlim(0.3, n_sources + 0.7)
    ax.set_ylim(0.5, 10.5)
    ax.set_yticks(range(1, 11))
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def draw_radar(ax, datasets, shared) -> None:
    n = len(DIMENSIONS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    for name, data, color in datasets:
        means = [float(np.mean(collect(data, shared, dim))) for dim in DIMENSIONS]
        means += means[:1]
        ax.plot(angles, means, color=color, linewidth=2, label=name)
        ax.fill(angles, means, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(SHORT_LABELS, fontsize=12)
    ax.tick_params(axis="x", pad=18)  # push dimension labels outward
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=11)
    ax.grid(alpha=0.4)
    ax.set_title("Mean score per dimension (radar)", pad=32, fontsize=15)
    # Source colors are already in the figure-level legend at the bottom.


def main() -> None:
    datasets = [(name, load(path), color) for name, path, color in SOURCES]
    shared = sorted(set.intersection(*(set(d.keys()) for _, d, _ in datasets)))
    sizes = " | ".join(f"{name}: {len(d)}" for name, d, _ in datasets)
    print(f"{sizes} | Shared: {len(shared)}")

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.0, 1.05],
        hspace=0.4, wspace=0.25,
    )

    # 7 violin panels: 4 on top row, 3 on bottom row. Radar takes the 8th cell.
    violin_axes = [fig.add_subplot(gs[0, c]) for c in range(4)]
    violin_axes += [fig.add_subplot(gs[1, c]) for c in range(3)]
    radar_ax = fig.add_subplot(gs[1, 3], projection="polar")

    for ax, dim in zip(violin_axes, DIMENSIONS):
        draw_violin(ax, dim, datasets, shared)
    draw_radar(radar_ax, datasets, shared)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black",
                      alpha=0.7, label=f"{name} (n={len(shared)})")
        for name, _, color in datasets
    ]
    handles += [
        plt.Line2D([0], [0], marker="o", color="white", markerfacecolor="white",
                   markeredgecolor="black", markersize=7, linestyle="None", label="median"),
        plt.Line2D([0], [0], marker="D", color="gray", markerfacecolor="gray",
                   markeredgecolor="black", markersize=7, linestyle="None", label="mean"),
        plt.Line2D([0], [0], color="black", linewidth=3, label="IQR"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"Score distributions across {len(shared)} shared papers "
        "— violins (equal area, per dimension) + radar (means)",
        fontsize=17,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {FIG_PATH}")

    for dim in DIMENSIONS:
        parts = [f"{name}={float(np.mean(collect(d, shared, dim))):.2f}"
                 for name, d, _ in datasets]
        print(f"{dim:45s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
