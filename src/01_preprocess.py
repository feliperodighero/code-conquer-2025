import pandas as pd
import re
import os

RAW_PATH = "src/data/raw/dataware.access.setembro2025.log"
PROCESSED_PATH = "src/data/processed/features.parquet"

def extract_features_from_log(log_line: str) -> dict:
    """
    Extrai features básicas de uma linha de log.
    Ajuste conforme o formato real dos logs fornecidos pelo professor.
    """

    match = re.match(r'(?P<ip>\S+) - - \[(?P<datetime>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\d+)', log_line)

    if match:
        data = match.groupdict()
    else:
        data = {}

    return {
        "len": len(log_line),  # comprimento da linha
        "num_digits": sum(c.isdigit() for c in log_line),
        "num_upper": sum(c.isupper() for c in log_line),
        "num_special": sum(1 for c in log_line if not c.isalnum() and c not in [" ", "\t"]),
        "has_sql": int(bool(re.search(r"(SELECT|DROP|INSERT|UPDATE|DELETE)", log_line, re.I))),
        "has_path_traversal": int(".." in log_line),
        "has_script_tag": int("<script" in log_line.lower()),
        "status": int(data.get("status", 0)),
        "size": int(data.get("size", 0)),
        "method_GET": int(data.get("method") == "GET"),
        "method_POST": int(data.get("method") == "POST"),
    }

def main():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        logs = f.readlines()

    features = []
    for line in logs:
        feat = extract_features_from_log(line)
        feat["raw_log"] = line.strip()  # salva a linha original
        features.append(feat)
    df = pd.DataFrame(features)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Features salvas em {PROCESSED_PATH}")

if __name__ == "__main__":
    main()
