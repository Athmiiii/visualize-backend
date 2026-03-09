import base64
import io
import logging

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

logger = logging.getLogger("backend.chart_generator")

CHART_TYPES = {"line", "bar", "scatter", "pie", "histogram", "area", "box", "heatmap"}


# ── Chart-configuration helpers ──────────────────────────────────────

def _is_numeric_dtype(dtype_name: str) -> bool:
    lowered = (dtype_name or "").lower()
    return any(token in lowered for token in ["int", "float", "double", "decimal", "number"])


def _is_datetime_dtype(dtype_name: str) -> bool:
    lowered = (dtype_name or "").lower()
    return "datetime" in lowered or "date" in lowered


def _resolve_column(name: str | None, columns: list) -> str | None:
    """Match a column name against actual DataFrame columns, case-insensitively."""
    if not name:
        return None
    if name in columns:
        return name
    name_stripped = name.strip().lower()
    for col in columns:
        if col.strip().lower() == name_stripped:
            return col
    return None


def _normalize_config_keys(config: dict) -> dict:
    """Map camelCase keys to snake_case for compatibility with JavaScript frontends."""
    key_map = {
        "chartType": "chart_type",
        "xAxis": "x_axis",
        "yAxis": "y_axis",
        "zAxis": "z_axis",
        "groupBy": "group_by",
        "topNGroups": "top_n_groups",
        "top_n": "top_n_groups",
        "aggregateBy": "aggregate_by",
        "type": "chart_type",
    }
    normalized = {}
    for k, v in config.items():
        mapped_key = key_map.get(k, k)
        
        if mapped_key not in normalized:
            normalized[mapped_key] = v
    return normalized


def _sanitize_final_config(config: dict, fallback: dict) -> dict:
    chart_type = (config.get("chart_type") or fallback.get("chart_type") or "bar").lower()
    if chart_type not in CHART_TYPES:
        chart_type = "bar"

    final = {
        "chart_type": chart_type,
        "x_axis": config.get("x_axis") or fallback.get("x_axis"),
        "y_axis": config.get("y_axis") if config.get("y_axis") is not None else fallback.get("y_axis"),
        "z_axis": config.get("z_axis") if config.get("z_axis") is not None else fallback.get("z_axis"),
        "group_by": config.get("group_by") if config.get("group_by") is not None else fallback.get("group_by"),
        "top_n_groups": config.get("top_n_groups") if config.get("top_n_groups") is not None else fallback.get("top_n_groups"),
        "aggregate_by": config.get("aggregate_by") if config.get("aggregate_by") is not None else fallback.get("aggregate_by"),
    }

    if final["chart_type"] == "histogram":
        final["y_axis"] = None
        final["aggregate_by"] = None

    return final


def _choose_best_config(
    schema: dict,
    sample_rows: list,
    user_config: dict,
    dataset_profile: dict | None,
    llm_final_config: dict | None,
) -> dict:
    
    user_config = _normalize_config_keys(user_config)
    if llm_final_config and isinstance(llm_final_config, dict):
        llm_final_config = _normalize_config_keys(llm_final_config)

    columns = list(schema.keys())
    numeric_cols = [col for col, dtype_name in schema.items() if _is_numeric_dtype(dtype_name)]
    datetime_cols = [col for col, dtype_name in schema.items() if _is_datetime_dtype(dtype_name)]
    categorical_cols = [
        col
        for col, dtype_name in schema.items()
        if col not in numeric_cols and col not in datetime_cols
    ]

    if not columns:
        return {
            "chart_type": "bar",
            "x_axis": None,
            "y_axis": None,
            "z_axis": None,
            "group_by": None,
            "top_n_groups": 10,
            "aggregate_by": "count",
        }

    profile = dataset_profile or {}
    column_stats = profile.get("column_stats", {}) if isinstance(profile, dict) else {}

    def coverage(col: str) -> float:
        stat = column_stats.get(col, {}) if isinstance(column_stats, dict) else {}
        value = stat.get("non_null_ratio")
        return float(value) if isinstance(value, (int, float)) else 1.0

    # ── Extract user preferences (with fuzzy column matching) ───────
    preferred_chart = (user_config.get("chart_type") or "").lower()
    user_x = _resolve_column(user_config.get("x_axis"), columns)
    user_y = _resolve_column(user_config.get("y_axis"), columns)
    user_z = _resolve_column(user_config.get("z_axis"), columns)
    user_group = _resolve_column(user_config.get("group_by"), columns)
    user_top_n = user_config.get("top_n_groups")
    user_agg = user_config.get("aggregate_by")

    valid_x = user_x is not None
    valid_y = user_y is not None
    valid_group = user_group is not None

    logger.info(
        "[_choose_best_config] preferred_chart=%s, user_x=%s (valid=%s), "
        "user_y=%s (valid=%s), raw_config_keys=%s",
        preferred_chart, user_x, valid_x, user_y, valid_y,
        list(user_config.keys()),
    )

    # ── For pie charts: auto-fill missing columns if possible ────────
    if preferred_chart == "pie":
        if not valid_x and valid_y:
            # Pick best categorical column for x, or first column
            user_x = categorical_cols[0] if categorical_cols else columns[0]
            valid_x = True
            logger.info("[pie] auto-filled x_axis=%s", user_x)
        if valid_x and not valid_y:
            # Pick first numeric column for y
            if numeric_cols:
                user_y = numeric_cols[0]
                valid_y = True
                logger.info("[pie] auto-filled y_axis=%s", user_y)
        if not valid_x and not valid_y:
            # Both missing: pick categorical + numeric if available
            if categorical_cols and numeric_cols:
                user_x = categorical_cols[0]
                user_y = numeric_cols[0]
                valid_x = valid_y = True
                logger.info("[pie] auto-filled x=%s, y=%s", user_x, user_y)
            elif len(columns) >= 2:
                user_x = columns[0]
                user_y = columns[1]
                valid_x = valid_y = True
                logger.info("[pie] auto-filled from first columns x=%s, y=%s", user_x, user_y)

    # ── Honor the user's chart type when columns are valid ───────────
    if preferred_chart in CHART_TYPES and valid_x:

        if preferred_chart == "scatter" and valid_y:
            return {
                "chart_type": "scatter",
                "x_axis": user_x,
                "y_axis": user_y,
                "z_axis": None,
                "group_by": user_group if valid_group else None,
                "top_n_groups": user_top_n or 10,
                "aggregate_by": None,
            }

        if preferred_chart == "line" and valid_y:
            return {
                "chart_type": "line",
                "x_axis": user_x,
                "y_axis": user_y,
                "z_axis": None,
                "group_by": user_group if valid_group else None,
                "top_n_groups": user_top_n or 10,
                "aggregate_by": user_agg or "mean",
            }

        if preferred_chart == "bar" and valid_y:
            return {
                "chart_type": "bar",
                "x_axis": user_x,
                "y_axis": user_y,
                "z_axis": None,
                "group_by": user_group if valid_group else None,
                "top_n_groups": user_top_n or 10,
                "aggregate_by": user_agg or "mean",
            }

        if preferred_chart == "pie" and valid_y:
            return {
                "chart_type": "pie",
                "x_axis": user_x,
                "y_axis": user_y,
                "z_axis": None,
                "group_by": None,
                "top_n_groups": user_top_n or 8,
                "aggregate_by": user_agg or "sum",
            }

        if preferred_chart == "histogram":
            return {
                "chart_type": "histogram",
                "x_axis": user_x,
                "y_axis": None,
                "z_axis": None,
                "group_by": None,
                "top_n_groups": user_top_n or 10,
                "aggregate_by": None,
            }

        if preferred_chart == "area" and valid_y:
            return {
                "chart_type": "area",
                "x_axis": user_x,
                "y_axis": user_y,
                "z_axis": None,
                "group_by": user_group if valid_group else None,
                "top_n_groups": user_top_n or 10,
                "aggregate_by": user_agg or "mean",
            }

        if preferred_chart == "box" and valid_y:
            return {
                "chart_type": "box",
                "x_axis": user_x,
                "y_axis": user_y,
                "z_axis": None,
                "group_by": None,
                "top_n_groups": user_top_n or 10,
                "aggregate_by": None,
            }

        if preferred_chart == "heatmap" and valid_y:
            z = user_z if (user_z and user_z in columns) else user_y
            return {
                "chart_type": "heatmap",
                "x_axis": user_x,
                "y_axis": user_y,
                "z_axis": z,
                "group_by": None,
                "top_n_groups": user_top_n or 10,
                "aggregate_by": user_agg or "mean",
            }

    # ── If the LLM provided a config with valid columns, use it ─────
    if llm_final_config and isinstance(llm_final_config, dict):
        llm_chart = (llm_final_config.get("chart_type") or "").lower()
        llm_x = llm_final_config.get("x_axis")
        llm_y = llm_final_config.get("y_axis")
        if llm_chart in CHART_TYPES and llm_x in columns:
            if llm_chart == "histogram" or (llm_y and llm_y in columns):
                fallback = {
                    "chart_type": "bar",
                    "x_axis": columns[0],
                    "y_axis": columns[-1] if len(columns) > 1 else columns[0],
                    "z_axis": None,
                    "group_by": None,
                    "top_n_groups": 10,
                    "aggregate_by": "count",
                }
                return _sanitize_final_config(llm_final_config, fallback)

    # ── Auto-detect fallback (no user preference or invalid axes) ────
    if datetime_cols and numeric_cols:
        return {
            "chart_type": "line",
            "x_axis": datetime_cols[0],
            "y_axis": numeric_cols[0],
            "z_axis": None,
            "group_by": None,
            "top_n_groups": 10,
            "aggregate_by": "mean",
        }

    if len(numeric_cols) >= 2:
        return {
            "chart_type": "scatter",
            "x_axis": numeric_cols[0],
            "y_axis": numeric_cols[1],
            "z_axis": None,
            "group_by": None,
            "top_n_groups": 10,
            "aggregate_by": None,
        }

    if categorical_cols and numeric_cols:
        return {
            "chart_type": "bar",
            "x_axis": categorical_cols[0],
            "y_axis": numeric_cols[0],
            "z_axis": None,
            "group_by": None,
            "top_n_groups": 10,
            "aggregate_by": "mean",
        }

    if numeric_cols:
        return {
            "chart_type": "histogram",
            "x_axis": numeric_cols[0],
            "y_axis": None,
            "z_axis": None,
            "group_by": None,
            "top_n_groups": 10,
            "aggregate_by": None,
        }

    fallback_col = columns[0]
    return {
        "chart_type": "bar",
        "x_axis": fallback_col,
        "y_axis": fallback_col,
        "z_axis": None,
        "group_by": None,
        "top_n_groups": 10,
        "aggregate_by": "count",
    }


def _contains_mismatched_chart_reference(text: str, final_chart_type: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    mentioned = {chart for chart in CHART_TYPES if chart in lowered}
    return bool(mentioned and (mentioned - {final_chart_type}))


def _aligned_interpretation(final_config: dict, llm_interp: dict | None) -> dict:
    chart_type = final_config.get("chart_type") or "bar"
    x_axis = final_config.get("x_axis") or "x-axis"
    y_axis = final_config.get("y_axis")
    agg = final_config.get("aggregate_by") or "value"

    if chart_type == "scatter":
        default_summary = f"This scatter chart compares {y_axis or 'values'} against {x_axis} to show how both variables move together."
        default_key = f"The key insight is the relationship pattern between {x_axis} and {y_axis or 'the measured metric'}."
        default_anomaly = "Look for isolated points far from the main cluster, as they may indicate unusual records."
        default_reco = f"Use this view to check whether changes in {x_axis} are associated with increases or decreases in {y_axis or 'the target metric'}."
    elif chart_type in {"line", "area"}:
        default_summary = f"This {chart_type} chart tracks {y_axis or 'the metric'} across {x_axis} to reveal overall direction and turning points over sequence or time."
        default_key = f"The key insight is how {y_axis or 'the metric'} trends as {x_axis} changes."
        default_anomaly = "Watch for sharp spikes or drops compared with nearby points."
        default_reco = f"Focus on trend shifts in {y_axis or 'the metric'} and investigate the periods where the slope changes notably."
    elif chart_type == "histogram":
        default_summary = f"This histogram shows the distribution of {x_axis}, highlighting where values are concentrated."
        default_key = f"The key insight is the most common range of {x_axis}."
        default_anomaly = "Look for long tails or isolated bins that suggest extreme values."
        default_reco = f"Use the distribution shape to set sensible thresholds or segments for {x_axis}."
    elif chart_type == "box":
        default_summary = f"This box chart summarizes spread and central tendency of {y_axis or 'the metric'} across {x_axis}."
        default_key = f"The key insight is the median and variability differences by {x_axis}."
        default_anomaly = "Points beyond whiskers indicate potential outliers."
        default_reco = f"Compare medians and spread across {x_axis} groups to prioritize stable high-performing segments."
    elif chart_type == "heatmap":
        default_summary = f"This heatmap displays {agg} {y_axis or 'values'} across {x_axis} combinations to reveal concentration patterns quickly."
        default_key = "The key insight is where higher-intensity cells cluster compared to the rest of the matrix."
        default_anomaly = "Cells with unexpectedly high or low intensity relative to neighboring cells are potential anomalies."
        default_reco = "Focus on the strongest and weakest cells to identify priority segments for deeper analysis."
    elif chart_type == "pie":
        default_summary = f"This pie chart shows each {x_axis} category's share of total {y_axis or 'value'}."
        default_key = f"The key insight is which {x_axis} categories dominate the total share."
        default_anomaly = "Very small slices may represent underrepresented categories worth investigating separately."
        default_reco = "Use this view for composition comparisons, then switch to bar for precise magnitude ranking if needed."
    else:  # bar
        default_summary = f"This bar chart compares {agg} {y_axis or 'values'} across {x_axis} categories for clear ranking."
        default_key = f"The key insight is which {x_axis} categories have the highest and lowest {y_axis or 'values'}."
        default_anomaly = "Large gaps between neighboring bars can indicate unusual category behavior."
        default_reco = f"Prioritize categories with consistently strong {y_axis or 'performance'} and investigate weaker ones."

    raw_summary = (llm_interp or {}).get("summary") or ""
    raw_key = (llm_interp or {}).get("key_insight") or ""
    raw_anomaly = (llm_interp or {}).get("anomalies") or ""
    raw_reco = (llm_interp or {}).get("recommendation") or ""

    use_defaults = any(
        _contains_mismatched_chart_reference(text, chart_type)
        for text in [raw_summary, raw_key, raw_anomaly, raw_reco]
    )

    if use_defaults:
        return {
            "summary": default_summary,
            "key_insight": default_key,
            "anomalies": default_anomaly,
            "recommendation": default_reco,
        }

    return {
        "summary": raw_summary or default_summary,
        "key_insight": raw_key or default_key,
        "anomalies": raw_anomaly or default_anomaly,
        "recommendation": raw_reco or default_reco,
    }


# ── Matplotlib rendering ─────────────────────────────────────────────

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
    # Pie requires positive values — drop zeros and negatives
    agg = agg[agg[y_col] > 0].copy()
    if agg.empty:
        raise ValueError(f"No positive values to display in pie chart for {y_col} grouped by {x_col}.")
    # Truncate long category labels for readability
    agg["_label"] = agg[x_col].astype(str).str[:30]
    wedges, texts, autotexts = ax.pie(
        agg[y_col],
        labels=agg["_label"],
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


def _bin_if_numeric(series: pd.Series, max_bins: int = 15) -> pd.Series:
    """Bin a continuous numeric column into equal-width ranges for pivot axes."""
    if not pd.api.types.is_numeric_dtype(series):
        return series
    nunique = series.nunique(dropna=True)
    if nunique <= max_bins:
        # Already low-cardinality (e.g. year as int) — keep as-is
        return series
    bins = min(max_bins, nunique)
    try:
        binned = pd.cut(series, bins=bins, duplicates="drop")
        # Convert intervals to readable strings like "10.0–20.0"
        return binned.apply(lambda iv: f"{iv.left:.1f}–{iv.right:.1f}" if pd.notna(iv) else iv)
    except Exception:
        return series


def _render_heatmap(ax, df, x_col, y_col, z_col, agg_method):
    work = df.copy()

    # Bin continuous numeric axes so pivot_table can group them
    x_label = x_col
    y_label = y_col
    work["__hm_x"] = _bin_if_numeric(work[x_col])
    work["__hm_y"] = _bin_if_numeric(work[y_col])

    # Limit cardinality so the heatmap stays readable
    MAX_CATS = 25
    x_top = work["__hm_x"].value_counts().head(MAX_CATS).index
    y_top = work["__hm_y"].value_counts().head(MAX_CATS).index
    filtered = work[work["__hm_x"].isin(x_top) & work["__hm_y"].isin(y_top)]

    pivot = filtered.pivot_table(
        index="__hm_y",
        columns="__hm_x",
        values=z_col,
        aggfunc=agg_method,
    )

    if pivot.empty:
        raise ValueError(
            f"Pivot table is empty for heatmap ({x_col} × {y_col} → {z_col}). "
            "Try different column combinations."
        )

    pivot = pivot.fillna(0)

    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(
        [str(c)[:20] for c in pivot.columns],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_yticklabels(
        [str(r)[:20] for r in pivot.index],
        fontsize=7,
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Heatmap — {z_col} by {x_label} and {y_label}", pad=15)


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

        # Validate z_col for heatmap
        if chart_type == "heatmap":
            actual_z = z_col or y_col
            if actual_z and actual_z not in df.columns:
                raise ValueError(f"Value column '{actual_z}' not found for heatmap.")

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

        if chart_type not in ("pie", "heatmap"):
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