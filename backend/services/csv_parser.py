import io
import pandas as pd

#CSV Validation
def parse_csv(file_bytes: bytes, filename: str) -> dict:
    try:
        df_KF = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not read '{filename}' as a CSV: {e}")

    if df_KF.empty:
        raise ValueError(f"No rows for'{filename}'")

    schema = {col: str(dtype) for col, dtype in df_KF.dtypes.items()}
    sample_rows = df_KF.tail(5).fillna("N/A").to_dict(orient="records")

    return {
        "dataframe": df_KF,
        "schema": schema,
        "sample_rows": sample_rows,
        "row_count": len(df_KF),
        "col_count": len(df_KF.columns),
    }