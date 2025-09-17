# src/preprocess.py

import pandas as pd
import re

# A função de extração de features permanece a mesma, ela é ótima!
def extract_features_from_log(log_line: str) -> dict:
    """
    Extrai features básicas de uma linha de log.
    """
    # Regex para o formato: IP - - [Data] "METHOD /path HTTP/1.1" STATUS SIZE
    match = re.match(r'(?P<ip>\S+) - - \[(?P<datetime>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+)', log_line)

    data = match.groupdict() if match else {}

    # Tratamento para size, que pode ser '-'
    size = data.get("size", "0")
    if not size.isdigit():
        size = "0"

    # Retornamos um dicionário com as features numéricas e os dados originais
    return {
        # Features para o modelo
        "len": len(log_line),
        "num_digits": sum(c.isdigit() for c in log_line),
        "num_upper": sum(c.isupper() for c in log_line),
        "num_special": sum(1 for c in log_line if not c.isalnum() and c not in [" ", "\t"]),
        "has_sql": int(bool(re.search(r"\b(SELECT|DROP|INSERT|UPDATE|DELETE)\b", log_line, re.I))),
        "has_path_traversal": int("../" in log_line),
        "has_script_tag": int("<script" in log_line.lower()),
        "status": int(data.get("status", 0)),
        "size": int(size),
        "method_GET": int(data.get("method") == "GET"),
        "method_POST": int(data.get("method") == "POST"),
        # Features para exibição no dashboard (não usadas pelo modelo)
        "ip": data.get("ip", "N/A"),
        "method": data.get("method", "N/A"),
        "raw_log": log_line.strip()
    }

# ✅ NOVA FUNÇÃO PRINCIPAL
def parse_log_to_dataframe(log_content: str) -> pd.DataFrame:
    """
    Recebe todo o conteúdo de um arquivo de log como string,
    processa cada linha e retorna um DataFrame completo.
    """
    # 1. Divide a string de conteúdo em uma lista de linhas
    logs = log_content.splitlines()

    # 2. Aplica a extração de features para cada linha
    features = [extract_features_from_log(line) for line in logs if line] # 'if line' ignora linhas vazias

    # 3. Converte a lista de dicionários em um DataFrame
    df = pd.DataFrame(features)

    return df

# A função main() e o if __name__ == "__main__": podem ser removidos ou mantidos
# apenas para testes independentes do script. Eles não serão usados pelo Streamlit.