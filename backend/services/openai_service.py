import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from services.chart_generator import (
    _normalize_config_keys,
    _choose_best_config,
    _aligned_interpretation,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger("backend.openai_service")

#System Prompt
SUGGESTIONS_SYSTEM_PROMPT = """
You are a data visualization expert working with municipal workers.
Always respond in JSON. Text outside the JSON is forbidden. 

Return a JSON object with this structure precisely:
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

#Validation Prompt
VALIDATION_SYSTEM_PROMPT = """
You are a data visualization expert and analyst for non-technical municipal workers.
Always respond in JSON. Never include any text outside the JSON.

IMPORTANT RULES:
- Never reject the configuration due to missing data in a selected pair.
- Always return the best available chart and best x/y pairing from the dataset.
- Avoid phrases like "invalid", "insufficient data", or "cannot analyze" unless the dataset has no usable columns at all.
- Prefer practical, readable choices that can be plotted immediately.

INTERPRETATION RULES (critical):
You are given real data (sample rows, column statistics with min/max/mean/median/std, row count).
Your "interpretation" MUST describe what the actual data shows, NOT what the chart "will show" or "can reveal".
- "summary": Describe the actual trend, pattern, or distribution you observe in the provided data.
  Use past/present tense ("The data shows...", "There is a positive correlation...").
  NEVER use future tense like "will visualize" or "will show".
  Include specific numbers from the column stats (e.g., "ranges from 2.1 to 98.5 with a mean of 45.3").
- "key_insight": State the single most significant pattern or relationship you found in the data.
  Describe HOW the variables actually affect each other, not that they "can inform decisions".
  Example: "Higher government spending is associated with 15-20% higher completion rates on average."
- "anomalies": Identify specific anomalies or outliers visible in the stats or sample rows.
  Reference actual values when possible. If none are apparent, say "No clear anomalies in the current sample."
- "recommendation": Give ONE specific, actionable next step based on the findings.
  Example: "Investigate the 5 countries with spending below 3% GDP that still exceed 90% completion rate."

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
    "summary": "2-3 sentences describing the observed pattern with specific numbers",
    "key_insight": "the single most important finding stating how variables affect each other",
    "anomalies": "specific unusual values or patterns found in the data",
    "recommendation": "one concrete action based on findings"
  }
}

Your final response must be understandable to any user within the municipality.
"""

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
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SUGGESTIONS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        result = json.loads(response.choices[0].message.content)

        if "suggestions" not in result:
            raise ValueError("Missing 'suggestions' key in response.")

        # Keep only suggestions with a correlation_score above 50%
        filtered = [
            s for s in result["suggestions"]
            if float(s.get("correlation_score", 0)) >= 0.5
        ]
        # Sort by correlation_score descending so the best match comes first
        filtered.sort(key=lambda s: float(s.get("correlation_score", 0)), reverse=True)
        result["suggestions"] = filtered
        result["total_options"] = len(filtered)

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
    # Normalize keys early so both the LLM prompt and _choose_best_config see snake_case
    user_config = _normalize_config_keys(user_config)
    logger.info(
        "[validate_chart_config] normalized user_config=%s",
        json.dumps(user_config, default=str),
    )

    user_message = (
        f"Dataset schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Dataset profile (includes min/max/mean/median/std for numeric columns):\n"
        f"{json.dumps(dataset_profile or {}, indent=2)}\n\n"
        f"Sample rows:\n{json.dumps(sample_rows, indent=2, default=str)}\n\n"
        f"User configuration:\n{json.dumps(user_config, indent=2)}\n\n"
        f"Analyze the actual data values above — use the column statistics and sample rows "
        f"to describe observed patterns, correlations, and anomalies with specific numbers. "
        f"Do NOT write generic descriptions of what the chart type does."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            temperature=0.3,
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

        logger.info(
            "[validate_chart_config] final best_config=%s",
            json.dumps(best_config, default=str),
        )

        interp = result.get("interpretation", {}) if isinstance(result, dict) else {}
        result["interpretation"] = _aligned_interpretation(best_config, interp)

        # Flag low-correlation matches (below 50%)
        correlation = float(result.get("correlation_score", 0))
        if correlation < 0.5:
            result["is_valid"] = False
            result["warning"] = (
                result.get("warning")
                or f"Low correlation score ({correlation:.0%}). "
                   "The selected chart may not represent the data well."
            )
        else:
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