import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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

Return a JSON object with exactly this structure:
{
  "is_valid": true or false,
  "correlation_score": 0.0 to 1.0,
  "validity_reason": "why this combination is valid or not",
  "warning": null or "warning message if combination is weak but allowed",
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


def validate_chart_config(schema: dict, sample_rows: list, user_config: dict) -> dict:
    user_message = (
        f"Dataset schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Sample rows:\n{json.dumps(sample_rows, indent=2, default=str)}\n\n"
        f"User configuration:\n{json.dumps(user_config, indent=2)}\n\n"
        f"Validate this configuration and return interpretation."
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

        interp = result.get("interpretation", {})
        result["interpretation"] = {
            "summary": interp.get("summary") or "Interpretation unavailable.",
            "key_insight": interp.get("key_insight") or "No key insight available.",
            "anomalies": interp.get("anomalies") or "No anomalies detected.",
            "recommendation": interp.get("recommendation") or "No recommendation available.",
        }

        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")
    except Exception as e:
        raise ValueError(f"OpenAI API error: {e}")