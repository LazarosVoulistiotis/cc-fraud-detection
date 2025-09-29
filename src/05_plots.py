#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Run on Powershell --> python .\src\05_plots.py --data ".\data\data_raw\creditcard.csv" --out ".\reports\figures\week4" --transparent

from pathlib import Path
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---- Theme ----
sns.set_theme(style="whitegrid", context="talk")
sns.set_palette("colorblind")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.unicode_minus": False,
    "legend.frameon": False,
    "legend.title_fontsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ---- Helpers ----
def save_fig(name: str, folder: Path, ext: str = "png", transparent: bool = False) -> Path:
    """Αποθήκευση τρέχοντος figure."""
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.{ext}"
    plt.savefig(out, bbox_inches="tight", transparent=transparent)
    print(f"[saved] {out}")
    return out

def annotate_bars(ax, show_count: bool = True, show_pct: bool = True,
                  fontsize: int = 9, color: str = "black"):
    """Counts & ποσοστά πάνω από μπάρες (για countplots)."""
    patches = [p for p in ax.patches if p.get_height() is not None]
    total = sum(p.get_height() for p in patches) if patches else 0
    if total == 0:
        return ax
    for p in patches:
        h = p.get_height()
        parts = []
        if show_count: parts.append(f"{int(h)}")
        if show_pct:   parts.append(f"({h/total:.2%})")
        ax.annotate("\n".join(parts),
                    (p.get_x()+p.get_width()/2., h),
                    ha="center", va="bottom", fontsize=fontsize, color=color)
    return ax

# =========================
# Plots
# =========================
def plot_class_distribution(df: pd.DataFrame, out_dir: Path, transparent: bool = False):
    # πιο έντονα χρώματα + καθαρή αντίθεση
    PALETTE = {"Legit (0)": "#1f77b4", "Fraud (1)": "#d62728"}

    df_plot = df.copy()
    df_plot["ClassLabel"] = df_plot["Class"].map({0: "Legit (0)", 1: "Fraud (1)"})

    fig = plt.figure(figsize=(6, 4))
    fig.patch.set_facecolor("white")  # σταθερό λευκό φόντο
    ax = sns.countplot(
        data=df_plot, x="ClassLabel", hue="ClassLabel",
        palette=PALETTE, legend=False
    )
    ax.set_title("Class Distribution (Legit vs Fraud)", fontsize=16, pad=10)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for bar in ax.patches:  # πιο έντονο περίγραμμα για να ξεχωρίζουν οι μπάρες
        bar.set_edgecolor("black")
        bar.set_linewidth(0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    annotate_bars(ax, fontsize=10)
    plt.tight_layout()
    save_fig("class_distribution", out_dir, transparent=transparent)
    plt.close()

def plot_amount_hist_linear(df: pd.DataFrame, out_dir: Path, transparent: bool = False):
    fig = plt.figure(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    plt.hist(
        df["Amount"],
        bins=80,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    plt.title("Distribution of Transaction Amounts (linear scale)", fontsize=16, pad=10)
    plt.xlabel("Transaction Amount (€)")
    plt.ylabel("Transaction Count")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    save_fig("amount_hist", out_dir, transparent=transparent)
    plt.close()

def plot_amount_hist_log(df: pd.DataFrame, out_dir: Path, transparent: bool = False) -> None:
    """Ιστόγραμμα Amount (log y, 99th pct cutoff)."""
    plt.figure(figsize=(7, 4))
    plt.hist(
        df["Amount"], bins=100,
        color="#1f77b4", edgecolor="black",
        alpha=0.85, linewidth=0.8
    )
    plt.yscale("log")
    plt.xlim(0, float(df["Amount"].quantile(0.99)))
    plt.title("Distribution of Transaction Amounts\n(log-scaled y-axis, 99th pct cutoff)", fontsize=14)
    plt.xlabel("Transaction Amount (€)")
    plt.ylabel("Transaction Count (log scale)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_fig("amount_hist_log", folder=out_dir, ext="png")
    plt.close()

from matplotlib.lines import Line2D

def plot_time_hist_by_class(df: pd.DataFrame, out_dir: Path, transparent: bool = False):
    # Χρησιμοποιούμε label στήλη για σταθερό legend
    PALETTE = {"Legit (0)": "#1f77b4", "Fraud (1)": "#d62728"}

    df_plot = df.copy()
    df_plot["ClassLabel"] = df_plot["Class"].map({0: "Legit (0)", 1: "Fraud (1)"})

    fig = plt.figure(figsize=(7.2, 4.5))
    fig.patch.set_facecolor("white")

    ax = sns.histplot(
        data=df_plot,
        x=df_plot["Time"] / 3600,
        bins=48,
        hue="ClassLabel",                 # <- string labels
        element="step",
        stat="count",
        common_norm=False,
        palette=PALETTE,
        linewidth=1.2,
        alpha=0.9,
        legend=False                      # <- φτιάχνουμε custom legend κάτω
    )

    ax.set_title("Transactions Over Time (0–48 hours)", fontsize=16, pad=10)
    ax.set_xlabel("Time (hours since first transaction)")
    ax.set_ylabel("Transaction Count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    # Custom, καθαρό legend
    handles = [
        Line2D([0], [0], color=PALETTE["Legit (0)"], lw=2, label="Legit (0)"),
        Line2D([0], [0], color=PALETTE["Fraud (1)"], lw=2, label="Fraud (1)"),
    ]
    leg = ax.legend(handles=handles, title="Class", frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)

    plt.tight_layout()
    save_fig("time_hist", out_dir, transparent=transparent)
    plt.close()

def plot_amount_by_class_overlay(df: pd.DataFrame, out_dir: Path, transparent: bool = False) -> None:
    """Κατανομή Amount ανά Class (normalized density overlay, 99th pct cutoff)."""
    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=df, x="Amount", hue="Class",
        bins=100, element="step", stat="density", common_norm=False,
        palette={0: "#1f77b4", 1: "#d62728"},  # πιο έντονα χρώματα
        linewidth=1.2
    )
    plt.xlim(0, float(df["Amount"].quantile(0.99)))
    plt.title("Normalized Distribution of Transaction Amounts (Legit vs Fraud)", fontsize=14)
    plt.xlabel("Transaction Amount (€)")
    plt.ylabel("Relative Frequency (Density)")
    plt.legend(title="Class", labels=["Legit (0)", "Fraud (1)"], frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_fig("amount_by_class_overlay", folder=out_dir, ext="png")
    plt.close()

def plot_amount_boxplot_log(df: pd.DataFrame, out_dir: Path, transparent: bool = False):
    PALETTE = {"Legit (0)": "#1f77b4", "Fraud (1)": "#d62728"}

    df_plot = df.copy()
    df_plot["ClassLabel"] = df_plot["Class"].map({0: "Legit (0)", 1: "Fraud (1)"})

    fig = plt.figure(figsize=(7.2, 4.5))
    fig.patch.set_facecolor("white")

    ax = sns.boxplot(
        data=df_plot,
        x="ClassLabel",
        y="Amount",
        hue="ClassLabel",
        palette=PALETTE,
        legend=False,
        linewidth=1.1,
        fliersize=2.5,  # λιγότερο «θόρυβος»
    )
    # κάνε τους outliers λίγο διαφανείς για να μην «μαυρίζει» η περιοχή
    for c in ax.collections:
        try:
            c.set_alpha(0.7)
        except Exception:
            pass

    plt.yscale("log")
    ax.set_title("Distribution of Transaction Amounts (Legit vs Fraud)", fontsize=16, pad=10)
    ax.set_xlabel("Class")
    ax.set_ylabel("Amount (€) [log scale]")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):,}" if y >= 1 else f"{y:g}"))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    save_fig("amount_by_class_box_log", out_dir, transparent=transparent)
    plt.close()

# --- Correlation: All numeric -------------------------------------------------
def plot_corr_heatmap_all(df: pd.DataFrame, out_dir: Path, transparent: bool = False):
    """
    Heatmap όλων των αριθμητικών μεταβλητών (Pearson).
    Σταθερή κλίμακα [-1, 1], τετράγωνο grid, λεπτές γραμμές για καθαρότητα.
    """
    num_df = df.select_dtypes(include="number")  # ασφαλές για pandas >=1.x
    corr_all = num_df.corr(method="pearson")

    fig = plt.figure(figsize=(11, 9))
    fig.patch.set_facecolor("white")

    ax = sns.heatmap(
        corr_all,
        cmap="coolwarm",
        vmin=-1, vmax=1, center=0,                 # συνεπής χρωματική κλίμακα
        square=True,
        linewidths=.3, linecolor="white",          # καθαρά όρια κελιών
        cbar_kws={"shrink": .85, "label": "Pearson r"}
    )
    ax.set_title("Correlation Heatmap (All Numeric Features)", pad=12, fontsize=16)
    plt.xticks(rotation=60, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig("corr_heatmap_full", out_dir, transparent=transparent)
    plt.close()


# --- Correlation: Time + V1–V28 + Amount + Class (masked lower triangle) -----
def plot_corr_heatmap_pca_masked(df: pd.DataFrame, out_dir: Path, transparent: bool = False):
    """
    Heatmap για Time, V1–V28, Amount, Class με τριγωνική μάσκα.
    Σημείωση: V1–V28 είναι PCA components → συχνά ασυσχέτιστα μεταξύ τους
    και οι συσχετίσεις δεν ερμηνεύονται όπως “raw” features.
    """
    pca_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    cols = ["Time"] + pca_cols + ["Amount", "Class"]
    sub = df[cols].copy()

    corr_sub = sub.corr(method="pearson")
    mask = np.triu(np.ones_like(corr_sub, dtype=bool))   # κρατάμε το κάτω τρίγωνο

    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor("white")

    ax = sns.heatmap(
        corr_sub, mask=mask,
        cmap="coolwarm", vmin=-1, vmax=1, center=0,
        square=True, linewidths=.3, linecolor="white",
        cbar_kws={"shrink": .85, "label": "Pearson r"}
    )
    ax.set_title("Correlation Heatmap (Time, V1–V28, Amount, Class)", pad=12, fontsize=16)
    plt.xticks(rotation=60, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig("corr_heatmap_pca_masked", out_dir, transparent=transparent)
    plt.close()


# --- Correlation vs target Class (barplot Top-N by |corr|) -------------------
def plot_corr_with_class_bar(df: pd.DataFrame, out_dir: Path, top_n: int = 15, transparent: bool = False):
    """
    Οριζόντιο barplot με τις Top-N μεταβλητές ως προς |corr| με το Class.
    Χρήσιμο για γρήγορο feature scouting (όχι για αιτιότητα).
    """
    corr_series = (
        df.select_dtypes("number")
          .corr(method="pearson")["Class"]
          .drop("Class")
          .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    fig = plt.figure(figsize=(9.5, 6.5))
    fig.patch.set_facecolor("white")
    ax = corr_series.head(top_n)[::-1].plot(
        kind="barh", color="#1f77b4", edgecolor="black", linewidth=0.8
    )
    ax.set_title(f"Top {top_n} |Correlation| with Class", pad=10, fontsize=16)
    ax.set_xlabel("Pearson r (abs. sorted)")
    ax.set_ylabel("Feature")
    # ετικέτες πάνω στις μπάρες
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.2f}",
                    (p.get_x() + p.get_width(), p.get_y() + p.get_height()/2),
                    va="center", ha="left", fontsize=10, xytext=(3, 0),
                    textcoords="offset points")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_fig("corr_with_class_top", out_dir, transparent=transparent)
    plt.close()


# --- Optional: Hierarchical clustermap ---------------------------------------
def plot_corr_clustermap_all(df: pd.DataFrame, out_dir: Path, transparent: bool = False):
    """
    Clustermap του correlation matrix για ομαδοποίηση χαρακτηριστικών.
    Προσοχή: βαρύ σε μεγάλα matrices—χρησιμοποίησέ το προαιρετικά.
    """
    num_df = df.select_dtypes(include="number")
    corr_all = num_df.corr(method="pearson")

    g = sns.clustermap(
        corr_all,
        cmap="coolwarm", vmin=-1, vmax=1, center=0,
        figsize=(11, 11), linewidths=.2,
        dendrogram_ratio=(.15, .05),
        cbar_kws={"shrink": .8, "label": "Pearson r"}
    )
    g.fig.suptitle("Correlation Clustermap (All Numeric Features)", y=1.02, fontsize=16)
    # το clustermap έχει δικό του save (g.savefig)
    g.savefig(out_dir / "corr_clustermap_full.png", dpi=150, bbox_inches="tight", transparent=transparent)
    plt.close()


# =========================
# Main (CLI)
# =========================
def main():
    parser = argparse.ArgumentParser(description="Generate EDA plots for credit card fraud dataset.")
    parser.add_argument("--data", required=True, type=str, help="Path to creditcard.csv")
    parser.add_argument("--out",  default="../../reports/figures/week4", type=str, help="Output folder for figures")
    parser.add_argument("--transparent", action="store_true", help="Export with transparent background")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir   = Path(args.out)

    if not data_path.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε το dataset: {data_path}")

    # Light preview
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    df.info(memory_usage="deep")
    print(f"Total missing values: {int(df.isnull().sum().sum())}")

    # Plots
    plot_class_distribution(df, out_dir, transparent=args.transparent)
    plot_amount_hist_linear(df, out_dir, transparent=args.transparent)
    plot_amount_hist_log(df, out_dir, transparent=args.transparent)
    plot_time_hist_by_class(df, out_dir, transparent=args.transparent)
    plot_amount_by_class_overlay(df, out_dir, transparent=args.transparent)
    plot_amount_boxplot_log(df, out_dir, transparent=args.transparent)
        # Correlation plots
    plot_corr_heatmap_all(df, out_dir, transparent=args.transparent)
    plot_corr_heatmap_pca_masked(df, out_dir, transparent=args.transparent)
    plot_corr_with_class_bar(df, out_dir, top_n=15, transparent=args.transparent)
    # προαιρετικό (αν θες και ιεραρχική ομαδοποίηση):
    plot_corr_clustermap_all(df, out_dir, transparent=args.transparent)

if __name__ == "__main__":
    main()
