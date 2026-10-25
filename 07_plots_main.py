"""
07_plots_main.py
================
Fix applied (Issue 3 companion):
  BEFORE: Training plot read raw training_curves.csv (single seed, no error bands).
  AFTER:  Loads training_curves_summary.csv when available and draws a shaded
          mean ± 1 std band representing variance across independent training seeds.
          Falls back gracefully to raw single-seed curves with a warning.
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("Images", exist_ok=True)

rs  = pd.read_csv("results/results_summary.csv")
ps  = pd.read_csv("results/results_per_sample.csv")
cfg = pickle.load(open("results/config.pkl", "rb"))
N   = cfg["N_DEVICES"]

# Load training curves — prefer summary CSV (mean ± std across seeds)
TC_SUMMARY = "results/training_curves_summary.csv"
TC_RAW     = "results/training_curves.csv"
if os.path.exists(TC_SUMMARY):
    tc_sum = pd.read_csv(TC_SUMMARY)
    HAS_ERRORBARS = True
    print("[07] Using training_curves_summary.csv — error bars reflect "
          f"{int(tc_sum['n_seeds'].max())} independent runs.")
else:
    tc_sum = None
    tc_raw = pd.read_csv(TC_RAW)
    HAS_ERRORBARS = False
    print("[07] WARNING: training_curves_summary.csv not found. "
          "Error bars will NOT be shown. Run 05_train.py with --seeds N first.")

METHODS = [
    "Always Local (M1)", "Always Offload (M2)", "No Semantic (M3)",
    "No Priority (M4)",  "Fixed Semantic (M5)", "Threshold-Based (M6)",
    "Random Policy (M7)","Zhao et al. (M8)",    "DQN Proposed (M9)",
]
COLORS = {
    "Always Local (M1)":    "#E8A020",
    "Always Offload (M2)":  "#3060C0",
    "No Semantic (M3)":     "#20A060",
    "No Priority (M4)":     "#9050C0",
    "Fixed Semantic (M5)":  "#C07020",
    "Threshold-Based (M6)": "#207080",
    "Random Policy (M7)":   "#808080",
    "Zhao et al. (M8)":     "#A02060",
    "DQN Proposed (M9)":    "#C03030",
}
HATCHES = {
    "Always Local (M1)":    "///",
    "Always Offload (M2)":  "\\\\\\",
    "No Semantic (M3)":     "xxx",
    "No Priority (M4)":     "+++",
    "Fixed Semantic (M5)":  "---",
    "Threshold-Based (M6)": "...",
    "Random Policy (M7)":   "///",
    "Zhao et al. (M8)":     "ooo",
    "DQN Proposed (M9)":    "***",
}
SHORT = {
    "Always Local (M1)":    "Always\nLocal",
    "Always Offload (M2)":  "Always\nOffload",
    "No Semantic (M3)":     "No\nSemantic",
    "No Priority (M4)":     "No\nPriority",
    "Fixed Semantic (M5)":  "Fixed\nSemantic",
    "Threshold-Based (M6)": "Threshold\nBased",
    "Random Policy (M7)":   "Random\nPolicy",
    "Zhao et al. (M8)":     "Zhao\net al.",
    "DQN Proposed (M9)":    "DQN\nProposed",
}
STYLES = {
    "Always Local (M1)":    (0, (1, 1)),
    "Always Offload (M2)":  "--",
    "No Semantic (M3)":     "-.",
    "No Priority (M4)":     (0, (3, 1, 1, 1)),
    "Fixed Semantic (M5)":  (0, (4, 2)),
    "Threshold-Based (M6)": (0, (2, 1, 2, 1)),
    "Random Policy (M7)":   ":",
    "Zhao et al. (M8)":     (0, (5, 1)),
    "DQN Proposed (M9)":    "-",
}

bar_labels  = [SHORT[m] for m in METHODS]
bar_colors  = [COLORS[m] for m in METHODS]
bar_hatches = [HATCHES[m] for m in METHODS]

def clean_ax(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8); s.set_color("black")
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", color="gray")
    ax.set_axisbelow(True)

def ema(x, a=0.92):
    out = []
    for v in x:
        out.append(v if not out else a * out[-1] + (1 - a) * v)
    return out

def savefig(name):
    plt.savefig(f"Images/{name}", dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved → Images/{name}")

def bar_chart(vals, ylabel, fname, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(bar_labels, vals, color=bar_colors, hatch=bar_hatches,
                  edgecolor="black", lw=1.0, width=0.55)
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + max(vals) * 0.025
        if val < 0.01: ypos = max(vals) * 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                fmt.format(val), ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12); ax.set_ylim(0, max(vals) * 1.28)
    clean_ax(ax); ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout(); savefig(fname)

acc_vals = [rs.loc[rs.method == m, "accuracy"].values[0] * 100 for m in METHODS]
lat_vals = [rs.loc[rs.method == m, "latency"].values[0] * 1000 for m in METHODS]
eng_vals = [rs.loc[rs.method == m, "energy"].values[0] * 10   for m in METHODS]
bw_vals  = [rs.loc[rs.method == m, "bw"].values[0] * 100      for m in METHODS]

print("[07] Generating main plots ...")
bar_chart(acc_vals, "Detection Accuracy (%)",
          f"IDS_bar_accuracy_N{N}.png",  "{:.1f}%")
bar_chart(lat_vals, "Process Time (ms)",
          f"IDS_bar_latency_N{N}.png")
bar_chart(eng_vals, "Energy Cons. (uAh)",
          f"IDS_bar_energy_N{N}.png",    "{:.1f}")
bar_chart(bw_vals,  "Bandwidth Util. (%)",
          f"IDS_bar_bandwidth_N{N}.png")

# 4-panel
fig, axs = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle(f"Comprehensive Comparison — All 9 Methods (N={N})",
             fontsize=13, fontweight="bold")
for ax, (vals, yl) in zip(axs.flat, [
    (lat_vals, "Process Time (ms)"), (eng_vals, "Energy Cons. (uAh)"),
    (bw_vals,  "Bandwidth Util. (%)"), (acc_vals, "Detection Accuracy (%)")]):
    bars = ax.bar(bar_labels, vals, color=bar_colors, hatch=bar_hatches,
                  edgecolor="black", lw=0.9, width=0.55)
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + max(vals) * 0.025
        if val < 0.01: ypos = max(vals) * 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")
    ax.set_ylabel(yl, fontsize=9); ax.set_ylim(0, max(vals) * 1.35)
    clean_ax(ax); ax.tick_params(axis="x", labelsize=7)
plt.tight_layout(); savefig(f"IDS_bar_4panel_N{N}.png")

# ── Training convergence with error bars ─────────────────────────────────────
# Error bars = variance across independent training seeds (not per-sample std).
# This matches the paper's description of Fig. 2 error bars.
TRAIN_C = {"DQN Proposed": "#C03030", "Zhao et al.": "#A02060"}
TRAIN_L = {"DQN Proposed": "-",       "Zhao et al.": "--"}

fig, ax = plt.subplots(figsize=(12, 5))

if HAS_ERRORBARS:
    for m in tc_sum["method"].unique():
        sub = tc_sum[tc_sum["method"] == m].sort_values("epoch")
        col = TRAIN_C.get(m, "gray")
        lw_ = 2.5 if "DQN" in m else 1.8
        ax.plot(sub["epoch"], sub["reward_mean"],
                color=col, ls=TRAIN_L.get(m, "-"), lw=lw_,
                marker="o", ms=3, label=m)
        # Shaded band = mean ± 1 std across independent seeds
        ax.fill_between(sub["epoch"],
                        sub["reward_mean"] - sub["reward_std"],
                        sub["reward_mean"] + sub["reward_std"],
                        alpha=0.20, color=col)
    n_seeds = int(tc_sum["n_seeds"].max())
    ax.set_title(
        f"Training Convergence  (mean ± std, {n_seeds} independent seeds)",
        fontsize=11)
else:
    # Fallback: plot raw single-seed curves without error bands
    for m in tc_raw["method"].unique():
        sub = tc_raw[tc_raw["method"] == m].sort_values("epoch")
        lw_ = 2.5 if "DQN" in m else 1.8
        ax.plot(sub["epoch"], sub["reward"],
                color=TRAIN_C.get(m, "gray"), ls=TRAIN_L.get(m, "-"),
                lw=lw_, marker="o", ms=3, label=m)
    ax.set_title(
        "Training Convergence (single seed — run 05_train.py --seeds N for error bars)",
        fontsize=10)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Avg Reward per Epoch", fontsize=12)
ax.legend(fontsize=10)
clean_ax(ax)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout(); savefig(f"IDS_training_N{N}.png")

# ── Per-sample line plots ─────────────────────────────────────────────────────
ex = list(range(len(ps)))
for metric, ylabel, fname in [
    ("accuracy", "Detection Accuracy",  f"IDS_accuracy_N{N}.png"),
    ("latency",  "Latency T_t",         f"IDS_latency_N{N}.png"),
    ("energy",   "Energy E_t",          f"IDS_energy_N{N}.png"),
]:
    fig, ax = plt.subplots(figsize=(13, 5))
    for m in METHODS:
        col = f"{m}_{metric}"
        if col not in ps.columns: continue
        lw_ = 2.5 if "DQN" in m else 1.4
        ax.plot(ex, ema(ps[col].values), color=COLORS[m],
                ls=STYLES[m], lw=lw_, label=SHORT[m].replace("\n", " "))
    dqn_col = f"DQN Proposed (M9)_{metric}"
    if dqn_col in ps.columns:
        ax.fill_between(ex, ema(ps[dqn_col].values),
                        alpha=0.10, color=COLORS["DQN Proposed (M9)"])
    ax.set_xlabel("Test Sample", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8, ncol=3)
    clean_ax(ax)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout(); savefig(fname)

print("[07] All main plots saved.")
