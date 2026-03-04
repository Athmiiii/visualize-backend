import io
import pandas as pd


def parse_csv(file_bytes: bytes, filename: str) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not read '{filename}' as CSV: {e}")

    if df.empty:
        raise ValueError(f"'{filename}' is empty — no rows found.")

    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample_rows = df.tail(5).fillna("N/A").to_dict(orient="records")

    return {
        "dataframe": df,
        "schema": schema,
        "sample_rows": sample_rows,
        "row_count": len(df),
        "col_count": len(df.columns),
    }