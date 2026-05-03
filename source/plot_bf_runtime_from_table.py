import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def table_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.arange(8, 19, dtype=float)
    greedy_us = np.array([11.6, 11.4, 11.0, 14.0, 14.0, 16.0, 24.0, 23.0, 23.0, 22.0, 24.0])
    bf_ms = np.array(
        [0.122, 0.217, 0.446, 0.951, 1.534, 3.224, 6.500, 14.022, 25.461, 58.162, 117.202]
    )
    return m, greedy_us, bf_ms


def plot_bf_scatter_same_scale(m: np.ndarray, greedy_s: np.ndarray, bf_s: np.ndarray, docs: pathlib.Path) -> pathlib.Path:
    b, ln_a = np.polyfit(m, np.log(bf_s), 1)
    a = float(np.exp(ln_a))
    dense_m = np.linspace(m.min(), m.max(), 200)
    fit_s = a * np.exp(b * dense_m)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.scatter(m, bf_s, color="tab:red", s=40, zorder=3, label="Observed (avg over trials)")
    ax.plot(
        dense_m,
        fit_s,
        color="hotpink",
        linewidth=2,
        zorder=2,
        label=f"Best Fit Trendline\n$t = {a:.2e} \\cdot e^{{{b:.3f}m}}$",
    )
    ax.scatter(m, greedy_s, color="tab:blue", s=40, zorder=3, label="Greedy (avg over trials)")

    ax.set_xlabel(r"Number of facilities $m$")
    ax.set_ylabel("Brute-force avg time (s)")
    ax.set_title(r"Brute-force runtime $s$ vs.\ Number of Facilities $m$")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", fontsize=9)

    out = docs / "bf_runtime_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_greedy_scatter(m: np.ndarray, greedy_us: np.ndarray, docs: pathlib.Path) -> pathlib.Path:
    """Greedy-only: time in microseconds vs m with linear least-squares trend."""
    slope, intercept = np.polyfit(m, greedy_us, 1)
    dense_m = np.linspace(m.min(), m.max(), 200)
    fit_us = slope * dense_m + intercept

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.scatter(m, greedy_us, color="tab:blue", s=40, zorder=3, label="Observed (avg over trials)")
    ax.plot(
        dense_m,
        fit_us,
        color="cornflowerblue",
        linewidth=2,
        zorder=2,
        label=f"Best Fit Trendline\n$t = {slope:.3f}m + {intercept:.2f}$ $\\mathrm{{\\mu s}}$",
    )

    ax.set_xlabel(r"Number of facilities $m$")
    ax.set_ylabel(r"Greedy avg time ($\mu$s)")
    ax.set_title(r"Greedy runtime vs.\ Number of Facilities $m$")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", fontsize=9)

    out = docs / "greedy_runtime_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dual_axis_combined(
    m: np.ndarray, greedy_us: np.ndarray, greedy_s: np.ndarray, bf_s: np.ndarray, docs: pathlib.Path
) -> pathlib.Path:
    b_bf, ln_a = np.polyfit(m, np.log(bf_s), 1)
    a_bf = float(np.exp(ln_a))
    dense_m = np.linspace(m.min(), m.max(), 200)
    fit_bf_s = a_bf * np.exp(b_bf * dense_m)

    slope_g, intercept_g = np.polyfit(m, greedy_us, 1)
    fit_g_us = slope_g * dense_m + intercept_g

    fig, ax_bf = plt.subplots(figsize=(8, 5), dpi=120)
    ax_g = ax_bf.twinx()

    ax_bf.scatter(m, bf_s, color="tab:red", s=40, zorder=3, label="BF observed (avg)")
    ax_bf.plot(
        dense_m,
        fit_bf_s,
        color="hotpink",
        linewidth=2,
        zorder=2,
        label=f"BF fit: $t = {a_bf:.2e} e^{{{b_bf:.3f}m}}$ s",
    )

    ax_g.scatter(m, greedy_us, color="tab:blue", s=40, zorder=3, label="Greedy observed (avg)")
    ax_g.plot(
        dense_m,
        fit_g_us,
        color="cornflowerblue",
        linewidth=2,
        linestyle="--",
        zorder=2,
        label=f"Greedy fit: $t = {slope_g:.3f}m + {intercept_g:.2f}$ $\\mu$s",
    )

    ax_bf.set_xlabel(r"Number of facilities $m$")
    ax_bf.set_ylabel("Brute-force avg time (s)", color="tab:red")
    ax_bf.tick_params(axis="y", labelcolor="tab:red")
    ax_g.set_ylabel(r"Greedy avg time ($\mu$s)", color="tab:blue")
    ax_g.tick_params(axis="y", labelcolor="tab:blue")

    ax_bf.set_title(r"Greedy vs.\ brute-force runtime vs.\ $m$")
    ax_bf.grid(True, linestyle="--", alpha=0.45)

    h1, l1 = ax_bf.get_legend_handles_labels()
    h2, l2 = ax_g.get_legend_handles_labels()
    ax_bf.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    out = docs / "bf_greedy_runtime_dual_axis.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    docs = repo_root() / "docs"
    m, greedy_us, bf_ms = table_data()
    greedy_s = greedy_us * 1e-6
    bf_s = bf_ms * 1e-3

    outputs = [
        plot_bf_scatter_same_scale(m, greedy_s, bf_s, docs),
        plot_greedy_scatter(m, greedy_us, docs),
        plot_dual_axis_combined(m, greedy_us, greedy_s, bf_s, docs),
    ]
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
