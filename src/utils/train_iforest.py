import pandas as pd
from sklearn.ensemble import IsolationForest


MODEL_FEATURES = [
    "len", "num_digits", "num_upper", "num_special",
    "has_sql", "has_path_traversal", "has_script_tag",
    "status", "size", "method_GET", "method_POST"
]


def run_isolation_forest(df: pd.DataFrame):
    """
    Recebe um DataFrame, treina o Isolation Forest e retorna as predições.
    """
    df_numeric = df[MODEL_FEATURES]

    model = IsolationForest(contamination="auto", random_state=42)

    preds = model.fit_predict(df_numeric)  # Retorna -1 para anomalia, 1 para normal

    return preds
