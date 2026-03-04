import json
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Dev 1 imports
from services.csv_parser import parse_csv
from services.data_cleaner import clean_data

# Dev 2 imports
from services.openai_service import get_chart_suggestions, validate_chart_config
from services.chart_generator import generate_chart

app = FastAPI(title="AI Data Visualizer", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_store: dict = {}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    file_bytes = await file.read()

    try:
        parsed = parse_csv(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    df_clean, cleaning_report = clean_data(parsed["dataframe"])

    session_id = file.filename
    _session_store[session_id] = {
        "df_clean": df_clean,
        "schema": {col: str(dtype) for col, dtype in df_clean.dtypes.items()},
        "sample_rows": df_clean.tail(5).fillna("N/A").to_dict(orient="records"),
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "row_count": len(df_clean),
        "col_count": len(df_clean.columns),
        "schema": {col: str(dtype) for col, dtype in df_clean.dtypes.items()},
        "sample_rows": df_clean.tail(5).fillna("N/A").to_dict(orient="records"),
        "cleaning_report": cleaning_report,
    }

@app.post("/suggestions")
async def get_suggestions(session_id: str = Form(...)):
    session = _session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload your CSV again.")

    try:
        result = get_chart_suggestions(
            schema=session["schema"],
            sample_rows=session["sample_rows"],
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-chart")
async def generate_chart_endpoint(
    session_id: str = Form(...),
    user_config: str = Form(...),
):
    session = _session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload your CSV again.")

    try:
        config_dict = json.loads(user_config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="user_config must be valid JSON.")

    try:
        validation = validate_chart_config(
            schema=session["schema"],
            sample_rows=session["sample_rows"],
            user_config=config_dict,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    final_config = validation.get("final_config", config_dict)
    chart_image_b64 = generate_chart(session["df_clean"], final_config)

    return {
        "chart_image": chart_image_b64,
        "is_valid": validation.get("is_valid", True),
        "warning": validation.get("warning"),
        "validity_reason": validation.get("validity_reason"),
        "alternative_suggestion": validation.get("alternative_suggestion"),
        "final_config": final_config,
        "interpretation": validation.get("interpretation", {}),
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "version": app.version}