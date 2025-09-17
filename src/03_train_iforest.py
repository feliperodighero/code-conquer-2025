import pandas as pd
from sklearn.ensemble import IsolationForest

PROCESSED_PATH = "src/data/processed/features.parquet"

def main():
    df = pd.read_parquet(PROCESSED_PATH)
    df_numeric = df.drop(columns=["raw_log"])


    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df_numeric)

    preds = model.predict(df_numeric)  # -1 = anomalia, 1 = normal
    df["anomaly"] = preds
    df.to_csv("src/data/processed/predictions_iforest.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    main()
