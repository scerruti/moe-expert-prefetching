#!/usr/bin/env python3
"""
Figures for phase_1/docs/experiments/k_scaling.md.

Two steps, so charts can be restyled without recomputing:
    python phase_1/scripts/k_scaling_figures.py data <collection output>/determinism
    python phase_1/scripts/k_scaling_figures.py plot

"data" writes figures/k_scaling_data.json; "plot" draws from it. Plotting needs
matplotlib, which isn't in requirements.txt: it's only used for these figures.
"""
import json
import sys
from pathlib import Path

import numpy as np

from k_scaling import load, needed_buffer, ranks
from k_scaling_theory import LAYER_GROUPS, fit_sqrt, gaussian_spacing, measure, simulate, slope, spacings, tail_exponent

OUT = Path(__file__).resolve().parents[1] / "docs" / "experiments" / "figures"
DATA = OUT / "k_scaling_data.json"
ALL_KS = list(range(1, 41))
COLLAPSE_KS = [1, 2, 3, 4, 6, 8, 12, 16, 24]
SYNTHETIC_KS = {8: [1, 2, 3, 4, 5, 6], 60: [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36, 40, 48],
                256: [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 200]}
REAL_ROUTING = {8: ("Mixtral 8x7B", 2), 60: ("Qwen1.5-MoE-A2.7B", 4), 256: ("DeepSeek-V3", 8)}


def top4_gap(probs):
    """Probability gap between the 4th and 5th expert, per row."""
    top = -np.sort(-probs, axis=1)
    return top[:, 3] - top[:, 4]


def near_tie_gaps(gap, changed, tokens, layers):
    """Split per-row gaps into all decisions, changed ones, and each token's first changed layer.

    gap: for each row, the smaller of the top-4 gaps in the two orderings compared.
    tokens: one hashable per row, identifying the token within this pair of orderings.
    """
    first = {}
    for i in np.flatnonzero(changed):
        token = tokens[i]
        if token not in first or layers[i] < layers[first[token]]:
            first[token] = i
    return gap, gap[changed], gap[list(first.values())]


def curve(values, grid):
    """Share of values at or below each grid point."""
    return (np.searchsorted(np.sort(values), grid, side="right") / len(values)).tolist()


def tail_curve(values, top_shares):
    """(value, share of values above it), for plotting a tail on log-log axes."""
    return {"value": np.quantile(values, 1 - np.array(top_shares)).tolist(), "share": list(top_shares)}


def compute(folder):
    folder = Path(folder)
    parts, gaps_all, gaps_changed, gaps_first = {}, [], [], []
    for dataset in ("gsm8k", "mbpp"):
        runs = sorted(folder.glob(f"run_*_{dataset}.parquet"))
        keys, reference = load(runs[0])
        lay = keys["layer_index"].to_numpy()
        tokens = list(zip(keys["sequence_id"].to_numpy(), keys["token_position"].to_numpy()))
        rank_a = ranks(reference)
        logs = np.log(np.maximum(reference, 1e-30))
        others = []
        for path in runs[1:]:
            other = load(path)[1]
            others.append(np.log(np.maximum(other, 1e-30)))
            changed = needed_buffer(rank_a, ranks(other), 4) > 0
            gap = np.minimum(top4_gap(reference), top4_gap(other))
            g_all, g_changed, g_first = near_tie_gaps(gap, changed, tokens, lay)
            gaps_all.append(g_all)
            gaps_changed.append(g_changed)
            gaps_first.append(g_first)
        parts[dataset] = (np.tile(lay, len(others)), np.concatenate([logs] * len(others)), np.concatenate(others))
    layers = np.concatenate([p[0] for p in parts.values()])
    a = np.concatenate([p[1] for p in parts.values()])
    b = np.concatenate([p[2] for p in parts.values()])
    data = {}

    def by_k(results):
        return {"k": list(results), "changed": [float(v[0]) for v in results.values()],
                "m95": [float(v[1]) for v in results.values()], "m99": [float(v[2]) for v in results.values()]}

    deep = layers >= 16
    data["qwen_all"] = by_k(measure(a, b, ALL_KS))
    data["qwen_deep"] = by_k(measure(a[deep], b[deep], ALL_KS))
    k16 = np.arange(1, 17)
    data["sqrt_fit_m95"] = [float(v) for v in fit_sqrt(k16, np.array(data["qwen_all"]["m95"][:16]))]

    gaps = spacings(a)
    spacing = gaps.mean(axis=0)
    gaussian = gaussian_spacing()
    sigma = float((spacing[:16] * gaussian[:16]).sum() / (gaussian[:16] ** 2).sum())
    delta = np.take_along_axis(b - a, np.argsort(-a, axis=1), axis=1)
    noise = np.abs(delta[:, :-1] - delta[:, 1:]).mean(axis=0)
    data["spacing"] = {"k": list(range(1, 60)), "measured": spacing.tolist(), "gaussian": (gaussian * sigma).tolist(),
                       "noise": noise.tolist(), "predicted_rate": (0.5 * noise / spacing).tolist()}

    shares = np.geomspace(0.5, 0.0005, 40).tolist()
    data["tail_k4"] = {}
    for name, (low, high) in LAYER_GROUPS.items():
        mask = (layers >= low) & (layers < high)
        values = np.abs(delta[mask, 3] - delta[mask, 4])
        values = values[values > 0]
        data["tail_k4"][name] = {**tail_curve(values, shares), "beta": tail_exponent(values)}

    data["collapse"] = {}
    for dataset, (lay, pa, pb) in parts.items():
        for name, (low, high) in LAYER_GROUPS.items():
            mask = (lay >= low) & (lay < high)
            data["collapse"][f"{dataset} | {name}"] = by_k(measure(pa[mask], pb[mask], COLLAPSE_KS))

    row_noise = (b - a).std(axis=1)
    k60 = SYNTHETIC_KS[60]
    data["noise_mix"] = {"k": k60, "qwen": by_k(measure(a, b, k60))["m95"]}
    for label, sizes in (("mixed", row_noise), ("same_median", np.array([np.median(row_noise)])),
                         ("same_mean", np.array([row_noise.mean()]))):
        data["noise_mix"][label] = by_k(simulate(60, k60, sigma, sizes))["m95"]
    data["experts"] = {}
    for n, ks in SYNTHETIC_KS.items():
        data["experts"][str(n)] = {**by_k(simulate(n, ks, sigma, row_noise)), "model": REAL_ROUTING[n][0], "real_k": REAL_ROUTING[n][1]}

    grid = np.geomspace(1e-6, 0.5, 120)
    data["near_ties"] = {"grid": grid.tolist(), "all": curve(np.concatenate(gaps_all), grid),
                         "changed": curve(np.concatenate(gaps_changed), grid), "first": curve(np.concatenate(gaps_first), grid),
                         "summary": {name: {"median": float(np.median(v)), "q99": float(np.quantile(v, 0.99)),
                                            "within_0.02": float((v <= 0.02).mean()), "within_0.005": float((v <= 0.005).mean()),
                                            "within_0.001": float((v <= 0.001).mean()),
                                            "exact_tie": float((v == 0).mean())}
                                     for name, v in (("all", np.concatenate(gaps_all)), ("changed", np.concatenate(gaps_changed)),
                                                     ("first", np.concatenate(gaps_first)))}}
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(data, indent=1))
    print(f"wrote {DATA}; near-ties {data['near_ties']['summary']}; "
          f"spacing slope k=1-16 {slope(k16, spacing[:16]):.2f}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": SECONDARY, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": SECONDARY,
    })
    data = json.loads(DATA.read_text())
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    percent = FuncFormatter(lambda v, _: f"{v * 100:g}%")

    def finish(fig, name, title, subtitle, top=0.82):
        fig.subplots_adjust(top=top)
        fig.text(0.02, 0.975, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
        fig.text(0.02, 0.975 - 0.27 / fig.get_figheight(), subtitle, fontsize=9.5, color=SECONDARY, va="top")
        fig.savefig(OUT / f"{name}.png", dpi=200)
        plt.close(fig)

    def log_x(ax):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(NullFormatter())

    def label(ax, x, y, text, **kw):
        ax.annotate(text, (x, y), fontsize=9, color=kw.pop("color", SECONDARY), **kw)

    # Figure 1: the question
    q, d = data["qwen_all"], data["qwen_deep"]
    a0, b0 = data["sqrt_fit_m95"]
    k = np.array(q["k"])
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.plot(k, k / 2, color=MUTED, lw=1.2)
    ax.plot(k, np.full(len(k), 2), color=MUTED, lw=1.2)
    k16 = k[k <= 16]
    ax.plot(k16, a0 + b0 * np.sqrt(k16), color=INK, lw=1.2)
    ax.plot(k, d["m95"], color=ORANGE, label="Layers 16–23")
    ax.plot(k, q["m95"], color=BLUE, label="All layers")
    i = q["k"].index(4)
    ax.plot([4], [q["m95"][i]], "o", ms=8, color=BLUE, mec=SURFACE, mew=1.5, zorder=5)
    ax.set_ylim(0, 6)
    ax.set_xlim(0, 41)
    ax.set_xlabel("Routing cut-off k")
    ax.set_ylabel("Buffer m covering 95% of changes")
    label(ax, 4, q["m95"][i], "k = 4: the only\nreal routing", xytext=(10, -26), textcoords="offset points")
    label(ax, 11.6, 5.55, "1.5k rule (m = k/2)", ha="left")
    label(ax, 40, 2, "k + 2 rule (m = 2)", ha="right", xytext=(0, -13), textcoords="offset points")
    label(ax, 16, a0 + b0 * 4, f"{a0:.2f} + {b0:.2f}√k\n(fit, k = 1–16)", xytext=(-30, -50), textcoords="offset points", color=INK)
    label(ax, 26, q["m95"][25], "All layers", ha="center", xytext=(0, -15), textcoords="offset points")
    label(ax, 26, d["m95"][25], "Layers 16–23", ha="center", xytext=(0, 8), textcoords="offset points")
    ax.legend(loc="lower right", ncol=2)
    finish(fig, "fig1_buffer_vs_k", "The buffer follows neither rule: it grows slowly, then levels off",
           "Qwen trial data. Beyond k = 4, a cut-off is applied to top-4 routing rather than run for real.")

    # Figure 2: steps 1 and 2
    s = data["spacing"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    k = np.array(s["k"])
    left.plot(k, s["gaussian"], color=ORANGE, label="Gaussian scores")
    left.plot(k, s["measured"], color=BLUE, label="Qwen, measured")
    ref = s["measured"][0] / k
    left.plot(k, ref, color=MUTED, lw=1.2)
    left.set_xscale("log"); left.set_yscale("log")
    left.xaxis.set_major_formatter(plain); left.xaxis.set_minor_formatter(NullFormatter())
    left.yaxis.set_major_formatter(plain); left.yaxis.set_minor_formatter(NullFormatter())
    left.set_ylim(0.008, 1)
    left.set_xlabel("Rank k")
    left.set_ylabel("Mean log-probability gap to next rank")
    left.set_title("Spacing shrinks as 1/k, down to mid-ranking", loc="left", fontsize=10, color=INK)
    label(left, 40, ref[39], "∝ 1/k", xytext=(-4, -14), textcoords="offset points", ha="right")
    left.annotate("Qwen smallest\nat k = 28", (28, s["measured"][27]), xytext=(2.2, 0.018), fontsize=9, color=SECONDARY,
                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    left.legend(loc="upper center")
    q = data["qwen_all"]
    n = len(q["k"])
    right.plot(q["k"], s["predicted_rate"][:n], color=ORANGE, label="Predicted: 0.5 × noise ÷ spacing")
    right.plot(q["k"], q["changed"], color=BLUE, label="Measured")
    right.set_xscale("log"); right.set_yscale("log")
    right.xaxis.set_major_formatter(plain); right.xaxis.set_minor_formatter(NullFormatter())
    right.yaxis.set_major_formatter(percent); right.yaxis.set_minor_formatter(NullFormatter())
    right.set_ylim(0.002, 0.2)
    right.set_xlabel("Cut-off k")
    right.set_ylabel("Share of decisions that change")
    right.set_title("The change rate follows, with nothing fitted", loc="left", fontsize=10, color=INK)
    right.legend(loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    finish(fig, "fig2_spacing_and_rate", "Steps 1 and 2: closer neighbours mean proportionally more changes",
           "Both panels use log axes. The prediction uses measured noise and spacing, with c = 0.5 from theory.", top=0.8)

    # Figure 3: the tail
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    colours = {"layers 0-7": BLUE, "layers 8-15": ORANGE, "layers 16-23": AQUA}
    for name, t in data["tail_k4"].items():
        ax.plot(t["value"], t["share"], color=colours[name], label=f"{name.capitalize()} (β ≈ {t['beta']:.1f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plain); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(percent); ax.yaxis.set_minor_formatter(NullFormatter())
    x = np.array([0.02, 0.11])
    ax.plot(x, 0.003 * (x / 0.02) ** -2, color=MUTED, lw=1.2)
    label(ax, 0.115, 0.003 * (0.115 / 0.02) ** -2, "β = 2 slope\n(would give √k)", xytext=(5, -4), textcoords="offset points")
    ax.annotate("Steps at multiples of 2⁻⁹:\nfp16 logit rounding", (0.0039, 0.13), xytext=(0.0024, 0.012), fontsize=9, color=SECONDARY,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("Size of the noise |δ|, ranks 4 and 5")
    ax.set_ylabel("Share of non-zero noise values above this size")
    ax.legend(loc="lower left")
    finish(fig, "fig3_noise_tail", "The noise has a heavy tail, but its steepness depends on the layer",
           "Qwen, k = 4. A straight line on these axes is a power law; β is measured from the 95th to 99.9th percentile.")

    # Figure 4: noise mix
    m = data["noise_mix"]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.plot(m["k"], m["same_median"], color=MUTED, lw=1.5, label="Synthetic, one noise size (median)")
    ax.plot(m["k"], m["same_mean"], color=AQUA, label="Synthetic, one noise size (mean)")
    ax.plot(m["k"], m["mixed"], color=ORANGE, label="Synthetic, Qwen's mix of noise sizes")
    ax.plot(m["k"], m["qwen"], color=BLUE, label="Qwen, measured")
    ax.set_ylim(0, 4.8); ax.set_xlim(0, 50)
    ax.set_xlabel("Routing cut-off k")
    ax.set_ylabel("Buffer m covering 95% of changes")
    at = m["k"].index(30)
    for key, text, dy in (("qwen", "Qwen", 7), ("mixed", "Mixed noise", 7), ("same_mean", "One size (mean)", 7), ("same_median", "One size (median)", -15)):
        label(ax, 30, m[key][at], text, ha="center", xytext=(0, dy), textcoords="offset points")
    ax.legend(loc="upper left", ncol=2)
    finish(fig, "fig4_noise_mix", "Buffer growth needs a mix of quiet and noisy rows",
           "A synthetic router with Gaussian scores and no cascades, varying only how noise size is distributed across rows.")

    # Figure 5: collapse
    groups = data["collapse"]
    colours = {"layers 0-7": BLUE, "layers 8-15": ORANGE, "layers 16-23": AQUA}
    shapes = {"gsm8k": "o", "mbpp": "s"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True)
    for name, g in groups.items():
        dataset, layer = name.split(" | ")
        style = dict(marker=shapes[dataset], color=colours[layer], ms=7, mec=SURFACE, mew=1, lw=0)
        left.plot(g["k"], g["m95"], **style)
        right.plot(g["changed"], g["m95"], **style)
    every_k = np.array([x for g in groups.values() for x in g["k"]], float)
    every_rate = np.array([x for g in groups.values() for x in g["changed"]])
    every_m = np.array([x for g in groups.values() for x in g["m95"]])
    for ax, x in ((left, every_k), (right, every_rate)):
        a1, b1 = fit_sqrt(x, every_m)
        xs = np.geomspace(x.min(), x.max(), 60)
        ax.plot(xs, a1 + b1 * np.sqrt(xs), color=MUTED, lw=1.2, zorder=1)
    log_x(left); log_x(right)
    right.xaxis.set_major_formatter(percent)
    left.set_xlabel("Cut-off k"); right.set_xlabel("Share of decisions that change")
    left.set_ylabel("Buffer m covering 95% of changes")
    left.set_title("Against k: layer groups separate", loc="left", fontsize=10, color=INK)
    right.set_title("Against change rate: they mostly line up", loc="left", fontsize=10, color=INK)
    handles = [Line2D([], [], marker="o", lw=0, color=c, ms=7, label=n.capitalize()) for n, c in colours.items()]
    handles += [Line2D([], [], marker=shapes[d], lw=0, color=MUTED, ms=7, label=t) for d, t in (("gsm8k", "GSM8K"), ("mbpp", "MBPP"))]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    finish(fig, "fig5_buffer_vs_rate", "The change rate predicts the buffer better than k does",
           "Qwen only. Each point is one k in one dataset and layer group. Grey curves are a + b√x fits.", top=0.8)

    # Figure 6: number of experts
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    colours = {"8": AQUA, "60": BLUE, "256": ORANGE}
    for n, e in data["experts"].items():
        x = np.array(e["k"], float)
        left.plot(x, e["m95"], color=colours[n], label=f"{n} experts")
        right.plot(x / int(n), e["m95"], color=colours[n])
        i = e["k"].index(e["real_k"])
        left.plot([e["real_k"]], [e["m95"][i]], "o", ms=8, color=colours[n], mec=SURFACE, mew=1.5, zorder=5)
        offset = {"8": (2, -15), "60": (3, -15), "256": (-4, 10)}[n]
        label(left, e["real_k"], e["m95"][i], f"{e['model'].split('-MoE')[0]}: k = {e['real_k']}", xytext=offset, textcoords="offset points",
              ha="right" if n == "256" else "left")
    log_x(left); log_x(right)
    right.xaxis.set_major_formatter(percent)
    left.set_ylim(0, 8.5)
    left.set_xlabel("Cut-off k"); right.set_xlabel("Cut-off as a share of experts (k/N)")
    left.set_ylabel("Buffer m covering 95% of changes")
    left.set_title("More experts, more room to grow", loc="left", fontsize=10, color=INK)
    right.set_title("The peak sits near the middle of the ranking", loc="left", fontsize=10, color=INK)
    left.legend(loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    finish(fig, "fig6_number_of_experts", "What the number of experts does in simulation",
           "Synthetic router: Gaussian scores with Qwen's noise. Dots mark each model's k. These are not measurements of those models.", top=0.8)

    # Figure 7: near-ties
    t = data["near_ties"]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.axvline(0.02, color=AXIS, lw=1); ax.axvline(0.005, color=AXIS, lw=1)
    ax.plot(t["grid"], t["all"], color=MUTED, lw=1.5, label="All decisions")
    ax.plot(t["grid"], t["changed"], color=BLUE, label="Changed decisions")
    ax.plot(t["grid"], t["first"], color=ORANGE, label="First flip per token")
    log_x(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.5f}".rstrip("0").rstrip(".")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_ylim(0, 1.03)
    ax.set_xlim(1e-5, 0.5)
    ties = t["summary"]
    label(ax, 1.1e-5, ties["first"]["exact_tie"], f"{ties['first']['exact_tie']:.0%} of first flips\nare exact ties", xytext=(0, -27), textcoords="offset points")
    label(ax, 1.1e-5, ties["changed"]["exact_tie"], f"{ties['changed']['exact_tie']:.0%} of changes", xytext=(0, -14), textcoords="offset points")
    ax.set_xlabel("Gap between 4th and 5th expert (probability), smaller of the two orderings")
    ax.set_ylabel("Share of decisions with a smaller gap")
    label(ax, 0.02, 0.06, "current window\n0.02", xytext=(5, 0), textcoords="offset points")
    label(ax, 0.005, 0.06, "proposed\n0.005", xytext=(-5, 0), textcoords="offset points", ha="right")
    ax.legend(loc="center right")
    finish(fig, "fig7_near_ties", "Changes happen at near-ties, but the 0.02 window barely tests that",
           "Qwen, k = 4. The 0.02 window contains 84% of all decisions; 0.005 contains half, and 99% of changed ones.")
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    if sys.argv[1] == "data":
        compute(sys.argv[2])
    else:
        plot()
