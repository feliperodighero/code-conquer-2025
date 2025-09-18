# utils/preprocess.py

import pandas as pd
import re

def parse_log_to_dataframe(log_content: str) -> pd.DataFrame:
    """
    Parseia o conteúdo de um log, extraindo features para ML e
    dados estruturados para visualização.
    """
    log_pattern = re.compile(
        r'(?P<ip>\S+) - - \[(?P<datetime>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+)'
    )

    data_list = []
    for line in log_content.strip().split('\n'):
        if not line:
            continue

        match = log_pattern.match(line)
        log_data = match.groupdict() if match else {}

        size = log_data.get("size", "0")
        if not size.isdigit():
            size = "0"

        # Dicionário com todas as features
        features = {
            # Features para o modelo de anomalia
            "len": len(line),
            "num_digits": sum(c.isdigit() for c in line),
            "num_upper": sum(c.isupper() for c in line),
            "num_special": sum(1 for c in line if not c.isalnum() and c not in [" ", "\t"]),
            "has_sql": int(bool(re.search(r"\b(SELECT|DROP|INSERT|UPDATE|DELETE)\b", line, re.I))),
            "has_path_traversal": int("../" in line),
            "has_script_tag": int("<script" in line.lower()),

            # Features básicas do log
            "ip": log_data.get("ip", "N/A"),
            "datetime_str": log_data.get("datetime", None),
            "method": log_data.get("method", "N/A"),
            "path": log_data.get("path", "N/A"),
            "status": int(log_data.get("status", 0)),
            "size": int(size),
            "raw_log": line.strip(),

            "method_GET": int(log_data.get("method") == "GET"),
            "method_POST": int(log_data.get("method") == "POST"),
        }
        data_list.append(features)

    if not data_list:
        return pd.DataFrame()

    df = pd.DataFrame(data_list)

    df['timestamp'] = pd.to_datetime(df['datetime_str'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')

    return df