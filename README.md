# 🚀 Code Conquer — LogAware

**LogAware** é uma aplicação interativa para **ingestão de logs e detecção de anomalias** em tempo real, construída com **Python, Streamlit e Machine Learning (Isolation Forest & Local Outlier Factor)**.
O sistema ajuda equipes de TI, DevOps e SRE a identificar padrões suspeitos em arquivos de log de maneira rápida, intuitiva e escalável.

---

## 📌 Funcionalidades
- Upload de arquivos `.log` ou `.txt`.
- Pré-processamento automático dos logs.
- Aplicação de **modelos de Machine Learning** para detectar anomalias:
  - Isolation Forest
  - Local Outlier Factor (LOF)
- Consolidação dos resultados de acordo com critérios selecionados pelo usuário.
- **Dashboards interativos** com Plotly:
  - Volume de logs por dia
  - Top 5 IPs com mais requisições
  - Distribuição de métodos HTTP
  - Requisições ao longo do tempo com média móvel
- **Visualização de anomalias** com paginação, busca e exportação para CSV.
- Tabela completa com todos os logs parseados, disponível para download.

---

## 🏗️ Arquitetura
A aplicação é dividida em três partes principais:
1. **Ingestão de logs** – Upload do arquivo e parsing com `utils/preprocess.py`.
2. **Modelos de detecção** – Execução dos algoritmos LOF e Isolation Forest (`utils/train_lof.py` e `utils/train_iforest.py`).
3. **Visualização e insights** – Interface Streamlit com gráficos, métricas e exportação de dados.

---

## ⚙️ Instalação

### Pré-requisitos
- Python 3.10+
- Ambiente virtual recomendado (`venv` ou `conda`)

### Passos
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/codeconquer-logaware.git
cd codeconquer-logaware

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.\.venv\Scripts\activate    # Windows

# Instale as dependências
pip install -r requirements.txt
