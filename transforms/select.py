def numeric_columns(df):
    """ColumnTransformer selector: names of every numeric column in a polars
    frame. Kept at module scope so a fitted pipeline stays picklable.
    """
    return [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
