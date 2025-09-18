import io
import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from utils.preprocess import parse_log_to_dataframe
from utils.train_iforest import run_isolation_forest
from utils.train_lof import run_lof

px.defaults.template = "simple_white"

st.set_page_config(
    page_title="Code Conquer · LogAware",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _format_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def _get_ts_column(df: pd.DataFrame):
    for c in ["timestamp", "time", "ts", "datetime", "date"]:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce")
            return ts
    return None

@st.cache_data(show_spinner=False)
def cached_parse(log_text: str) -> pd.DataFrame:
    return parse_log_to_dataframe(log_text)

@st.cache_data(show_spinner=False)
def cached_models(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['anomaly_lof'] = run_lof(out)
    out['anomaly_iforest'] = run_isolation_forest(out)
    return out

def plot_top_ips(df: pd.DataFrame):
    if "ip" not in df or df["ip"].dropna().empty:
        return None

    s = df["ip"].value_counts().nlargest(5)
    top_df = s.reset_index()
    top_df.columns = ["ip", "count"]

    fig = px.bar(
        top_df.sort_values("count"),
        x="count",
        y="ip",
        orientation="h",
        text="count",
        labels={"ip": "IP", "count": "Requisições"},
        title="Top 5 IPs por Requisições",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(categoryorder="array", categoryarray=top_df.sort_values("count")["ip"].tolist()),
    )
    return fig

def plot_http_methods(df: pd.DataFrame):
    if 'method' not in df:
        return None
    method_counts = df['method'].value_counts()
    if method_counts.empty:
        return None
    fig = px.pie(
        method_counts,
        names=method_counts.index,
        values=method_counts.values,
        hole=0.35,
        title='Métodos HTTP'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def _pick_rule(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if pd.isna(start) or pd.isna(end):
        return "1D"
    delta = end - start
    days = delta.days + delta.seconds/86400
    if days <= 3:
        return "1H"    
    elif days <= 180:
        return "1D"    
    else:
        return "1W"    

def plot_requests_over_time(df: pd.DataFrame, date_min=None, date_max=None):
    ts = _get_ts_column(df)
    if ts is None:
        return None

    tmp = df.copy()
    tmp["__ts__"] = ts
    tmp = tmp.dropna(subset=["__ts__"])
    if tmp.empty:
        return None

    if date_min is not None:
        tmp = tmp[tmp["__ts__"] >= pd.to_datetime(date_min)]
    if date_max is not None:
        tmp = tmp[tmp["__ts__"] <= pd.to_datetime(date_max)]

    if tmp.empty:
        return None

    start = tmp["__ts__"].min()
    end = tmp["__ts__"].max()

    rule = _pick_rule(start, end)

    series = (
        tmp.set_index("__ts__")
           .resample(rule)
           .size()
           .rename("count")
           .reset_index()
    )

    window_map = {"1H": 3, "1D": 3, "1W": 3}
    w = window_map.get(rule, 3)
    series["rolling"] = series["count"].rolling(w, min_periods=1).mean()

    fig = px.line(
        series,
        x="__ts__",
        y="count",
        labels={"__ts__": "Data", "count": "Requisições"},
        title=f"Requisições ao longo do tempo ({rule})",
    )
    fig.add_scatter(
        x=series["__ts__"],
        y=series["rolling"],
        mode="lines",
        name=f"Média móvel ({w}×{rule})",
    )

    fig.update_xaxes(
        tickangle=-45,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=30, label="30d", step="day", stepmode="backward"),
                dict(count=90, label="90d", step="day", stepmode="backward"),
                dict(step="all", label="Tudo"),
            ])
        ),
        rangeslider=dict(visible=True),
        type="date",
    )

    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      h1, h2 {letter-spacing: 0.2px}
      .stDataFrame {border: 1px solid #eee; border-radius: 12px;}
      .pager-center {text-align:center; color:#7a7a7a; margin-top: .5rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Code Conquer — LogAware")
st.caption("Upload do arquivo .log, detecção de anomalias e insights de forma simples e objetiva.")

st.sidebar.header("Arquivo de log")
uploaded_file = st.sidebar.file_uploader("Arraste e solte ou escolha um arquivo .log", type=["log", "txt"])  # txt por segurança

with st.sidebar.expander("Preferências de visualização", expanded=False):
    crit = st.selectbox(
        "Critério de anomalia",
        (
            "LOF ou Isolation Forest (mais sensível)",
            "Ambos (mais conservador)",
            "Somente LOF",
            "Somente Isolation Forest",
        ),
        help="Como consolidar a coluna `is_anomaly`."
    )
    items_per_page = st.slider("Itens por página", 5, 50, 12)
    show_raw = st.toggle("Mostrar coluna de log bruto (raw_log)", value=False)

if not uploaded_file:
    st.info("Envie um arquivo `.log` na barra lateral para iniciar a análise.")
    st.stop()

if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
    st.session_state.current_file = uploaded_file.name
    st.session_state.page_number = 0

try:
    raw_bytes = uploaded_file.getvalue()
    size_info = _format_bytes(len(raw_bytes))
    log_text = raw_bytes.decode("utf-8", errors="replace")
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.stop()

with st.status("Pré-processando…", expanded=False):
    df = cached_parse(log_text)

with st.status("Aplicando modelos…", expanded=False):
    df = cached_models(df)

ts_detectado = _get_ts_column(df)
if ts_detectado is not None:
    df["timestamp"] = ts_detectado

if crit == "LOF ou Isolation Forest (mais sensível)":
    df['is_anomaly'] = (df['anomaly_lof'] == -1) | (df['anomaly_iforest'] == -1)
elif crit == "Ambos (mais conservador)":
    df['is_anomaly'] = (df['anomaly_lof'] == -1) & (df['anomaly_iforest'] == -1)
elif crit == "Somente LOF":
    df['is_anomaly'] = (df['anomaly_lof'] == -1)
else:  # Somente Isolation Forest
    df['is_anomaly'] = (df['anomaly_iforest'] == -1)

left, mid, right = st.columns(3)
left.metric("Total de Requisições", f"{len(df):,}")
mid.metric("Total de Anomalias", f"{int(df['is_anomaly'].sum()):,}")
right.metric("Taxa de Anomalia", f"{(df['is_anomaly'].mean()*100 if len(df) else 0):.2f}%")

st.caption(f"**Arquivo:** {uploaded_file.name} · **Tamanho:** {size_info}")

tab_overview, tab_anoms, tab_logs = st.tabs(["Visão Geral", "Anomalias", "Tabela completa"])

with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_top_ips(df)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Sem dados de IP para o ranking Top 5.")
    with col2:
        fig2 = plot_http_methods(df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Sem dados de método HTTP para o gráfico.")

    with st.expander("Filtros rápidos", expanded=False):
        methods = sorted(df['method'].dropna().unique().tolist()) if 'method' in df else []
        statuses = sorted(df['status'].dropna().unique().tolist()) if 'status' in df else []
        s1, s2 = st.columns(2)
        with s1:
            pick_methods = st.multiselect("Métodos", methods)
        with s2:
            pick_status = st.multiselect("Status", statuses)
        filtered = df.copy()
        if pick_methods:
            filtered = filtered[filtered['method'].isin(pick_methods)]
        if pick_status:
            filtered = filtered[filtered['status'].isin(pick_status)]
        st.dataframe(
            filtered.head(200),
            use_container_width=True,
            height=350,
        )

with tab_anoms:
    anom = df.loc[df['is_anomaly']].copy()
    if anom.empty:
        st.success("Nenhuma anomalia detectada com o critério atual.")
    else:
        anom['detected_by'] = anom.apply(
            lambda r: 'LOF & IForest' if (r['anomaly_lof'] == -1 and r['anomaly_iforest'] == -1)
            else ('LOF' if r['anomaly_lof'] == -1 else 'IForest'), axis=1
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            q = st.text_input("Pesquisar (IP, método, status, path…) ")
        with c2:
            st.download_button(
                "Baixar CSV das anomalias",
                data=anom.to_csv(index=False).encode('utf-8'),
                file_name=f"anomalias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if q:
            cols = [c for c in ['raw_log', 'ip', 'method', 'status', 'path'] if c in anom.columns]
            mask = pd.Series(False, index=anom.index)
            for c in cols:
                mask |= anom[c].astype(str).str.contains(q, case=False, na=False)
            anom = anom[mask]

        if 'page_number' not in st.session_state:
            st.session_state.page_number = 0
        total_items = len(anom)
        total_pages = max(1, (total_items - 1) // items_per_page + 1)
        st.session_state.page_number = min(st.session_state.page_number, total_pages - 1)

        start = st.session_state.page_number * items_per_page
        end = min(start + items_per_page, total_items)

        display_cols = [c for c in ['ip', 'method', 'status', 'size', 'detected_by', 'anomaly_lof', 'anomaly_iforest', 'raw_log'] if c in anom.columns]
        if not show_raw and 'raw_log' in display_cols:
            display_cols.remove('raw_log')

        st.dataframe(
            anom.iloc[start:end][display_cols],
            use_container_width=True,
            height=420,
        )

        cprev, cinfo, cnext = st.columns([1,2,1])
        with cprev:
            if st.button('⬅️ Anterior', disabled=(st.session_state.page_number == 0), use_container_width=True):
                st.session_state.page_number -= 1
                st.rerun()
        with cinfo:
            st.markdown(f"<div class='pager-center'>Página {st.session_state.page_number + 1} de {total_pages}</div>", unsafe_allow_html=True)
        with cnext:
            if st.button('Próxima ➡️', disabled=(st.session_state.page_number >= total_pages - 1), use_container_width=True):
                st.session_state.page_number += 1
                st.rerun()

with tab_logs:
    st.caption("Prévia das primeiras linhas parseadas (máx. 1000 linhas)")
    st.dataframe(df.head(1000), use_container_width=True, height=500)
    st.download_button(
        "Baixar CSV completo",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f"logs_parseados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
