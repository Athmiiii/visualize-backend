import base64
import io

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = [
    "#89b4fa", "#a6e3a1", "#fab387", "#f38ba8",
    "#cba6f7", "#94e2d5", "#eba0ac", "#f9e2af",
]

DARK_BG = "#1e1e2e"
DARK_SURFACE = "#313244"
TEXT_COLOR = "#cdd6f4"
SPINE_COLOR = "#45475a"
GRID_COLOR = "#45475a"


def _apply_dark_theme(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def _aggregate(df: pd.DataFrame, x_col: str, y_col: str, method: str) -> pd.DataFrame:
    agg_map = {"mean": "mean", "sum": "sum", "count": "count", "median": "median"}
    agg_func = agg_map.get(method, "mean")
    return (
        df.groupby(x_col)[y_col]
        .agg(agg_func)
        .reset_index()
        .sort_values(x_col)
    )


def _filter_top_groups(df: pd.DataFrame, group_col: str, y_col: str, top_n: int) -> pd.DataFrame:
    top_groups = (
        df.groupby(group_col)[y_col]
        .sum()
        .nlargest(top_n)
        .index
    )
    return df[df[group_col].isin(top_groups)]


def _set_labels(ax, chart_type: str, x_col: str, y_col: str, agg_method: str, title_prefix: str = ""):
    ax.set_title(
        f"{title_prefix}{chart_type.capitalize()} — {y_col} vs {x_col}",
        pad=15, fontsize=12,
    )
    if chart_type != "pie":
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(f"{y_col} ({agg_method})", fontsize=10)


def _render_line(ax, df, x_col, y_col, group_col, top_n, agg_method):
    if group_col:
        filtered = _filter_top_groups(df, group_col, y_col, top_n)
        for i, (name, group) in enumerate(filtered.groupby(group_col)):
            agg = _aggregate(group, x_col, y_col, agg_method)
            ax.plot(
                agg[x_col], agg[y_col],
                label=str(name),
                color=PALETTE[i % len(PALETTE)],
                linewidth=2, marker="o", markersize=4
            )
        ax.legend(facecolor=DARK_SURFACE, labelcolor=TEXT_COLOR, fontsize=8)
    else:
        agg = _aggregate(df, x_col, y_col, agg_method)
        ax.plot(
            agg[x_col], agg[y_col],
            color=PALETTE[0],
            linewidth=2, marker="o", markersize=4
        )


def _render_bar(ax, df, x_col, y_col, group_col, top_n, agg_method):
    if group_col:
        filtered = _filter_top_groups(df, group_col, y_col, top_n)
        pivot = (
            filtered
            .pivot_table(index=x_col, columns=group_col, values=y_col, aggfunc=agg_method)
            .head(top_n)
        )
        pivot.plot(kind="bar", ax=ax, color=PALETTE[:len(pivot.columns)])
        ax.legend(facecolor=DARK_SURFACE, labelcolor=TEXT_COLOR, fontsize=8)
    else:
        agg = _aggregate(df, x_col, y_col, agg_method).head(top_n)
        bars = ax.bar(agg[x_col], agg[y_col], color=PALETTE[0])
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f"{height:.1f}",
                ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=8
            )


def _render_scatter(ax, df, x_col, y_col, group_col, top_n):
    if group_col:
        filtered = _filter_top_groups(df, group_col, y_col, top_n)
        for i, (name, group) in enumerate(filtered.groupby(group_col)):
            ax.scatter(
                group[x_col], group[y_col],
                label=str(name),
                color=PALETTE[i % len(PALETTE)],
                alpha=0.7, s=50
            )
        ax.legend(facecolor=DARK_SURFACE, labelcolor=TEXT_COLOR, fontsize=8)
    else:
        ax.scatter(df[x_col], df[y_col], color=PALETTE[0], alpha=0.7, s=50)


def _render_pie(ax, df, x_col, y_col, top_n):
    agg = _aggregate(df, x_col, y_col, "sum").nlargest(top_n, y_col)
    wedges, texts, autotexts = ax.pie(
        agg[y_col],
        labels=agg[x_col],
        autopct="%1.1f%%",
        colors=PALETTE[:len(agg)],
        textprops={"color": TEXT_COLOR, "fontsize": 8},
        startangle=90,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5},
    )
    for autotext in autotexts:
        autotext.set_color(DARK_BG)
        autotext.set_fontweight("bold")


def _render_histogram(ax, df, x_col):
    ax.hist(
        df[x_col].dropna(),
        bins=20,
        color=PALETTE[0],
        edgecolor=DARK_BG,
        alpha=0.85
    )
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(f"Distribution of {x_col}", pad=15, fontsize=12)


def _render_area(ax, df, x_col, y_col, group_col, top_n, agg_method):
    if group_col:
        filtered = _filter_top_groups(df, group_col, y_col, top_n)
        for i, (name, group) in enumerate(filtered.groupby(group_col)):
            agg = _aggregate(group, x_col, y_col, agg_method)
            ax.fill_between(
                agg[x_col], agg[y_col],
                alpha=0.4,
                color=PALETTE[i % len(PALETTE)],
                label=str(name)
            )
            ax.plot(
                agg[x_col], agg[y_col],
                color=PALETTE[i % len(PALETTE)],
                linewidth=1.5
            )
        ax.legend(facecolor=DARK_SURFACE, labelcolor=TEXT_COLOR, fontsize=8)
    else:
        agg = _aggregate(df, x_col, y_col, agg_method)
        ax.fill_between(agg[x_col], agg[y_col], alpha=0.4, color=PALETTE[0])
        ax.plot(agg[x_col], agg[y_col], color=PALETTE[0], linewidth=2)


def _render_box(ax, df, x_col, y_col, top_n):
    top_cats = df[x_col].value_counts().head(top_n).index
    filtered = df[df[x_col].isin(top_cats)]

    groups = [
        filtered[filtered[x_col] == cat][y_col].dropna().values
        for cat in top_cats
    ]

    bp = ax.boxplot(
        groups,
        labels=[str(cat) for cat in top_cats],
        patch_artist=True,
        medianprops={"color": TEXT_COLOR, "linewidth": 2},
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)


def _render_heatmap(ax, df, x_col, y_col, z_col, agg_method):
    try:
        pivot = df.pivot_table(
            index=y_col,
            columns=x_col,
            values=z_col,
            aggfunc=agg_method
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(pivot.index, fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Heatmap — {z_col} by {x_col} and {y_col}", pad=15)
    except Exception:
        _render_bar(ax, df, x_col, z_col or y_col, None, 10, agg_method)


def generate_chart(df: pd.DataFrame, chart_decision: dict) -> str:
    chart_type = chart_decision.get("chart_type", "bar").lower()
    x_col = chart_decision.get("x_axis")
    y_col = chart_decision.get("y_axis")
    group_col = chart_decision.get("group_by")
    z_col = chart_decision.get("z_axis")
    top_n = int(chart_decision.get("top_n_groups") or 5)
    agg_method = chart_decision.get("aggregate_by") or "mean"

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_theme(fig, ax)

    try:
        for col in filter(None, [x_col, y_col]):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found.")

        if chart_type == "line":
            _render_line(ax, df, x_col, y_col, group_col, top_n, agg_method)
            _set_labels(ax, chart_type, x_col, y_col, agg_method)

        elif chart_type == "bar":
            _render_bar(ax, df, x_col, y_col, group_col, top_n, agg_method)
            _set_labels(ax, chart_type, x_col, y_col, agg_method)

        elif chart_type == "scatter":
            _render_scatter(ax, df, x_col, y_col, group_col, top_n)
            _set_labels(ax, chart_type, x_col, y_col, agg_method)

        elif chart_type == "pie":
            _render_pie(ax, df, x_col, y_col, top_n)

        elif chart_type == "histogram":
            _render_histogram(ax, df, x_col)

        elif chart_type == "area":
            _render_area(ax, df, x_col, y_col, group_col, top_n, agg_method)
            _set_labels(ax, chart_type, x_col, y_col, agg_method)

        elif chart_type == "box":
            _render_box(ax, df, x_col, y_col, top_n)

        elif chart_type == "heatmap":
            _render_heatmap(ax, df, x_col, y_col, z_col or y_col, agg_method)

        else:
            _render_bar(ax, df, x_col, y_col, group_col, top_n, agg_method)
            _set_labels(ax, "bar", x_col, y_col, agg_method)

        plt.xticks(rotation=45)
        plt.tight_layout()

    except Exception as exc:
        ax.text(
            0.5, 0.5,
            f"Chart error:\n{exc}",
            transform=ax.transAxes,
            ha="center", va="center",
            color="#f38ba8",
            fontsize=11,
            wrap=True,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")