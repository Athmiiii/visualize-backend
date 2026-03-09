import pandas as pd


def clean_data(df: pd.DataFrame) -> tuple:
    report = []
    df = df.copy()

    #Dropping all the unnecessary columns 
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)
        report.append({
            "type": "empty_columns_dropped",
            "message": f"Dropped {len(empty_cols)} empty column(s): {empty_cols}"
        })

    #Duplicates removal
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        report.append({
            "type": "duplicates_removed",
            "message": f"Removed {duplicate_count} duplicate row or rows."
        })

    #Effective formatting
    string_cols = df.select_dtypes(include="object").columns.tolist()
    whitespace_fixed = []
    casing_fixed = []

    for col in string_cols:
        before = df[col].copy()
        df[col] = df[col].str.strip()
        if not df[col].equals(before):
            whitespace_fixed.append(col)

        before = df[col].copy()
        df[col] = df[col].str.lower()
        if not df[col].equals(before):
            casing_fixed.append(col)

    if whitespace_fixed:
        report.append({
            "type": "whitespace_stripped",
            "message": f"Stripped whitespace in column(s): {whitespace_fixed}"
        })

    if casing_fixed:
        report.append({
            "type": "casing_normalised",
            "message": f"Normalised text casing in: {', '.join(casing_fixed)}"
        })

    #Convert numeric strings
    converted_cols = []
    for col in string_cols:
        if col not in df.columns:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        success_rate = converted.notna().sum() / len(df)

        if success_rate > 0.5:
            df[col] = converted
            converted_cols.append(col)

    if converted_cols:
        report.append({
            "type": "dtype_converted",
            "message": f"Converted to numeric: {converted_cols}"
        })

    #Fill null values
    null_report = []

    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            df[col].fillna(fill_value, inplace=True)
            null_report.append(
                f"'{col}': {null_count} null(s) filled with median ({fill_value:.2f})"
            )
        else:
            mode_series = df[col].mode()
            fill_value = mode_series[0] if not mode_series.empty else "unknown"
            df[col].fillna(fill_value, inplace=True)
            null_report.append(
                f"'{col}': {null_count} null(s) filled with mode ('{fill_value}')"
            )

    if null_report:
        report.append({
            "type": "nulls_filled",
            "message": "Filled missing values — " + " | ".join(null_report)
        })

    #Flag negative values
    negative_report = []
    for col in df.select_dtypes(include="number").columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            negative_report.append(f"'{col}': {negative_count} negative value(s)")

    if negative_report:
        report.append({
            "type": "negative_values_flagged",
            "message": "Negative values detected (not removed) — " + " | ".join(negative_report)
        })

    if not report:
        report.append({
            "type": "no_issues_found",
            "message": "Dataset looks clean. No issues detected."
        })

    return df, report