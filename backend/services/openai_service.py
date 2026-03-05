import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


CHART_TYPES = {"line", "bar", "scatter", "pie", "histogram", "area", "box", "heatmap"}


SUGGESTIONS_SYSTEM_PROMPT = """
You are a data visualization expert working with non-technical municipal workers.
Always respond in JSON. Never include any text outside the JSON.

Return a JSON object with exactly this structure:
{
  "suggestions": [
    {
      "option_id": 1,
      "chart_type": "one of: line, bar, scatter, pie, histogram, area, box, heatmap",
      "x_axis": "column name",
      "y_axis": "column name or null for histogram",
      "z_axis": "column name or null — only used for heatmap as the value column",
      "group_by": "column name or null",
      "aggregate_by": "one of: mean, sum, count, median or null",
      "top_n_groups": integer or null,
      "correlation_score": 0.0 to 1.0,
      "why_this_works": "one sentence in plain English",
      "what_to_look_for": "one sentence telling user what insight to expect"
    }
  ],
  "total_options": integer
}
"""


VALIDATION_SYSTEM_PROMPT = """
You are a data visualization expert and analyst for non-technical municipal workers.
Always respond in JSON. Never include any text outside the JSON.

IMPORTANT RULES:
- Never reject the configuration due to missing data in a selected pair.
- Always return the best available chart and best x/y pairing from the dataset.
- Avoid phrases like "invalid", "insufficient data", or "cannot analyze" unless the dataset has no usable columns at all.
- Prefer practical, readable choices that can be plotted immediately.

Return a JSON object with exactly this structure:
{
    "is_valid": true,
  "correlation_score": 0.0 to 1.0,
    "validity_reason": "why this final recommended combination is appropriate",
    "warning": null or "short quality warning",
  "alternative_suggestion": null or {
    "chart_type": "...", "x_axis": "...", "y_axis": "...", "group_by": null
  },
  "final_config": {
    "chart_type": "one of: line, bar, scatter, pie, histogram, area, box, heatmap",
    "x_axis": "column name",
    "y_axis": "column name or null",
    "z_axis": "column name or null",
    "group_by": "column name or null",
    "top_n_groups": integer or null,
    "aggregate_by": "one of: mean, sum, count, median or null"
  },
  "interpretation": {
    "summary": "2-3 sentences",
    "key_insight": "single most important finding",
    "anomalies": "unusual patterns or outliers",
    "recommendation": "one clear action"
  }
}
"""


def _is_numeric_dtype(dtype_name: str) -> bool:
    lowered = (dtype_name or "").lower()
    return any(token in lowered for token in ["int", "float", "double", "decimal", "number"])


def _is_datetime_dtype(dtype_name: str) -> bool:
    lowered = (dtype_name or "").lower()
    return "datetime" in lowered or "date" in lowered


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

    preferred_chart = (user_config.get("chart_type") or "").lower()
    user_x = user_config.get("x_axis")
    user_y = user_config.get("y_axis")

    if (
        preferred_chart == "scatter"
        and user_x in numeric_cols
        and user_y in numeric_cols
        and coverage(user_x) >= 0.6
        and coverage(user_y) >= 0.6
    ):
        return {
            "chart_type": "scatter",
            "x_axis": user_x,
            "y_axis": user_y,
            "z_axis": None,
            "group_by": None,
            "top_n_groups": 10,
            "aggregate_by": None,
        }

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
    llm_candidate = llm_final_config or {}
    fallback = {
        "chart_type": "bar",
        "x_axis": fallback_col,
        "y_axis": fallback_col,
        "z_axis": None,
        "group_by": None,
        "top_n_groups": 10,
        "aggregate_by": "count",
    }
    return _sanitize_final_config(llm_candidate, fallback)


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
        default_summary = f"This pie chart shows each {x_axis} category’s share of total {y_axis or 'value'}."
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


def _get_unique_counts(sample_rows: list) -> dict:
    unique_counts = {}
    if sample_rows:
        for col in sample_rows[0].keys():
            values = [str(row[col]) for row in sample_rows if row.get(col) is not None]
            unique_counts[col] = len(set(values))
    return unique_counts


def get_chart_suggestions(schema: dict, sample_rows: list) -> dict:
    unique_counts = _get_unique_counts(sample_rows)

    user_message = (
        f"Dataset schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Unique value counts:\n{json.dumps(unique_counts, indent=2)}\n\n"
        f"Sample rows:\n{json.dumps(sample_rows, indent=2, default=str)}\n\n"
        f"Return 3 to 6 chart suggestions covering different chart types."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SUGGESTIONS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        result = json.loads(response.choices[0].message.content)

        if "suggestions" not in result:
            raise ValueError("Missing 'suggestions' key in response.")

        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")
    except Exception as e:
        raise ValueError(f"OpenAI API error: {e}")


def validate_chart_config(
    schema: dict,
    sample_rows: list,
    user_config: dict,
    dataset_profile: dict | None = None,
) -> dict:
    user_message = (
        f"Dataset schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Dataset profile:\n{json.dumps(dataset_profile or {}, indent=2)}\n\n"
        f"Sample rows:\n{json.dumps(sample_rows, indent=2, default=str)}\n\n"
        f"User configuration:\n{json.dumps(user_config, indent=2)}\n\n"
        f"Return the best chart configuration for this data and add interpretation."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        result = json.loads(response.choices[0].message.content)

        best_config = _choose_best_config(
            schema=schema,
            sample_rows=sample_rows,
            user_config=user_config,
            dataset_profile=dataset_profile,
            llm_final_config=result.get("final_config") if isinstance(result, dict) else None,
        )

        interp = result.get("interpretation", {}) if isinstance(result, dict) else {}
        result["interpretation"] = _aligned_interpretation(best_config, interp)

        result["is_valid"] = True
        result["warning"] = result.get("warning") if result.get("warning") else None
        result["validity_reason"] = (
            result.get("validity_reason")
            or "Configuration optimized to the most suitable chart and axes for this dataset."
        )
        result["alternative_suggestion"] = None
        result["final_config"] = best_config

        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")
    except Exception as e:
        raise ValueError(f"OpenAI API error: {e}")