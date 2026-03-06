import asyncio
import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from services.csv_parser import parse_csv
from services.data_cleaner import clean_data
from services.openai_service import get_chart_suggestions, validate_chart_config
from services.chart_generator import generate_chart

app = FastAPI(title="AI Data Visualizer", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_store: dict = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")


def _resolve_session(session_id: str):
    if not session_id:
        return None

    if session_id in _session_store:
        return _session_store[session_id]

    normalized = session_id.strip().lower()

    if normalized.endswith(".csv"):
        without_ext = normalized[:-4]
    else:
        without_ext = normalized

    for key, value in _session_store.items():
        key_norm = key.strip().lower()
        if key_norm == normalized:
            return value
        if key_norm.endswith(".csv") and key_norm[:-4] == without_ext:
            return value
        if key_norm == f"{without_ext}.csv":
            return value

    return None


#Endpoint 1: CSV Uploader
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
 
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

 
    file_bytes = await file.read()
    filename = file.filename

    async def event_stream():
        try:
            parsed = parse_csv(file_bytes, filename)
            df_clean, cleaning_report = clean_data(parsed["dataframe"])

            schema = {col: str(dtype) for col, dtype in df_clean.dtypes.items()}
            sample_rows = df_clean.tail(5).where(df_clean.notna(), other=None).to_dict(orient="records")

            column_stats = {}
            for col in df_clean.columns:
                stat = {
                    "dtype": str(df_clean[col].dtype),
                    "non_null_ratio": round(float(df_clean[col].notna().mean()), 4),
                    "unique_count": int(df_clean[col].nunique(dropna=True)),
                }
                if df_clean[col].dtype.kind in ("i", "f"): 
                    desc = df_clean[col].describe()
                    stat.update({
                        "min": round(float(desc["min"]), 4),
                        "max": round(float(desc["max"]), 4),
                        "mean": round(float(desc["mean"]), 4),
                        "median": round(float(desc["50%"]), 4),
                        "std": round(float(desc["std"]), 4),
                    })
                column_stats[col] = stat

            dataset_profile = {
                "row_count": int(len(df_clean)),
                "column_stats": column_stats,
            }

            
            _session_store[filename] = {
                "df_clean":    df_clean,
                "schema":      schema,
                "sample_rows": sample_rows,
                "dataset_profile": dataset_profile,
            }

            
            yield json.dumps({
                "event":           "cleaning_done",
                "session_id":      filename,
                "cleaning_report": cleaning_report,
                "schema":          schema,
                "row_count":       len(df_clean),
            }) + "\n"

       
            await asyncio.sleep(0)

           
            loop = asyncio.get_event_loop()
            suggestions = await loop.run_in_executor(
                None,
                get_chart_suggestions,
                schema,
                sample_rows,
            )

            
            yield json.dumps({
                "event":         "suggestions_ready",
                "suggestions":   suggestions.get("suggestions", []),
                "total_options": suggestions.get("total_options", 0),
            }) + "\n"

        except Exception as exc:
            logger.error(f"Stream error: {exc}", exc_info=True)
            yield json.dumps({
                "event":   "error",
                "message": str(exc),
            }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


#Endpoint 2: Chart Generation JSON

@app.post("/generate-chart")
async def generate_chart_endpoint(
    session_id:  str = Form(...),
    user_config: str = Form(...),
):
  
    session = _resolve_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session not found for the provided session_id. "
                "Re-upload the CSV to create a fresh session, then call /generate-chart "
                "with the new session_id returned by /upload-csv. "
                "If the backend auto-reloaded, previous in-memory sessions were cleared."
            )
        )

    try:
        config_dict = json.loads(user_config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="user_config must be valid JSON.")

    logger.info(
        "[generate-chart] session_id=%s, raw user_config=%s",
        session_id, json.dumps(config_dict, default=str),
    )

    try:
        validation = validate_chart_config(
            schema=session["schema"],
            sample_rows=session["sample_rows"],
            user_config=config_dict,
            dataset_profile=session.get("dataset_profile"),
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    final_config = validation.get("final_config", config_dict)
    chart_image_b64 = generate_chart(session["df_clean"], final_config)

    return {
        "chart_image":            chart_image_b64,
        "is_valid":               validation.get("is_valid", True),
        "warning":                validation.get("warning"),
        "validity_reason":        validation.get("validity_reason"),
        "alternative_suggestion": validation.get("alternative_suggestion"),
        "final_config":           final_config,
        "interpretation":         validation.get("interpretation", {}),
    }