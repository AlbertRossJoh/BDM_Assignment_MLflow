def numeric_columns(df):
    return [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
