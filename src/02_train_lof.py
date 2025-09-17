import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

PROCESSED_PATH = "src/data/processed/features.parquet"

def main():
    df = pd.read_parquet(PROCESSED_PATH)

    df_numeric = df.drop(columns=["raw_log"])

    model = LocalOutlierFactor(n_neighbors=100, novelty=False)
    preds = model.fit_predict(df_numeric)

    df["anomaly"] = preds  # -1 = anomalia, 1 = normal
    df.to_csv("src/data/processed/predictions_lof.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    main()
