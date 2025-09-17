import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


MODEL_FEATURES = [
    "len", "num_digits", "num_upper", "num_special",
    "has_sql", "has_path_traversal", "has_script_tag",
    "status", "size", "method_GET", "method_POST"
]

def run_lof(df: pd.DataFrame):
    """
    Recebe um DataFrame, treina o Local Outlier Factor e retorna as predições.
    """
    df_numeric = df[MODEL_FEATURES]

    model = LocalOutlierFactor(n_neighbors=100, novelty=False, contamination="auto")

    preds = model.fit_predict(df_numeric)

    return preds
