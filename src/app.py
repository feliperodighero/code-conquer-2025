import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocess import parse_log_to_dataframe
from utils.train_iforest import run_isolation_forest
from utils.train_lof import run_lof


def plot_top_ips(df):
    top_ips = df['ip'].value_counts().nlargest(10)
    fig = px.bar(top_ips, x=top_ips.index, y=top_ips.values, title='Top 10 IPs por número de requisições',
                 labels={'index': 'Endereço IP', 'y': 'Número de Requisições'})
    return fig

def plot_http_methods(df):
    method_counts = df['method'].value_counts()
    fig = px.pie(method_counts, names=method_counts.index, values=method_counts.values,
                 title='Distribuição de Métodos HTTP', hole=.3)
    return fig


st.set_page_config(layout="wide")
st.title("🚀 Code Conquer - Análise de Logs com IA")
st.markdown("Faça o upload do seu arquivo `.log` para detectar anomalias e visualizar insights.")

st.sidebar.header("1. Envie seu arquivo de log")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo", type=['log'])

if uploaded_file is not None:
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        if 'page_number' in st.session_state:
            st.session_state.page_number = 0

    log_content = uploaded_file.getvalue().decode("utf-8")

    with st.spinner('Passo 1/3: Pré-processando o arquivo de log...'):
        df = parse_log_to_dataframe(log_content)

    with st.spinner('Passo 2/3: Aplicando modelos de IA...'):
        df['anomaly_lof'] = run_lof(df)
        df['anomaly_iforest'] = run_isolation_forest(df)
        df['is_anomaly'] = (df['anomaly_lof'] == -1) | (df['anomaly_iforest'] == -1)

    st.success("Análise concluída com sucesso!")

    st.header("Dashboard de Análise")
    total_requests = len(df)
    total_anomalies = df['is_anomaly'].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Requisições", f"{total_requests:,}")
    col2.metric("Total de Anomalias Detectadas", f"{total_anomalies:,}")
    if total_requests > 0:
        col3.metric("Taxa de Anomalia", f"{(total_anomalies / total_requests) * 100:.2f}%")

    st.markdown("---")
    st.header("Visualizações Gerais")
    col1_fig, col2_fig = st.columns(2)
    with col1_fig:
        st.plotly_chart(plot_top_ips(df), use_container_width=True)
    with col2_fig:
        st.plotly_chart(plot_http_methods(df), use_container_width=True)

    st.markdown("---")
    st.header("Anomalias Detectadas")

    anomalies_df = df.loc[df['is_anomaly'], ['raw_log', 'ip', 'method', 'status', 'size', 'anomaly_lof', 'anomaly_iforest']].copy()

    if not anomalies_df.empty:
        anomalies_df['detected_by'] = anomalies_df.apply(
            lambda row: 'LOF & IForest' if row['anomaly_lof'] == -1 and row['anomaly_iforest'] == -1
            else 'LOF' if row['anomaly_lof'] == -1
            else 'IForest',
            axis=1
        )

        if 'page_number' not in st.session_state:
            st.session_state.page_number = 0

        items_per_page = 10
        total_items = len(anomalies_df)
        total_pages = max(1, (total_items -1) // items_per_page + 1)


        start_index = st.session_state.page_number * items_per_page
        end_index = min(start_index + items_per_page, total_items)

        st.dataframe(anomalies_df.iloc[start_index:end_index])

        col1, col2, col3 = st.columns([3, 3, 1])

        with col1:
            if st.button('⬅️ Anterior', disabled=(st.session_state.page_number == 0), use_container_width=True):
                st.session_state.page_number -= 1
                st.rerun()

        with col3:
            if st.button('Próxima ➡️', disabled=(st.session_state.page_number >= total_pages - 1), use_container_width=True):
                st.session_state.page_number += 1
                st.rerun()

        with col2:
            st.markdown(f"<p style='text-align: center; color: grey; margin-top: 10px'>Página {st.session_state.page_number + 1} de {total_pages}</p>", unsafe_allow_html=True)

    else:
        st.info("Nenhuma anomalia foi detectada pelos modelos.")

else:
    st.info("Aguardando o envio de um arquivo `.log` para iniciar a análise.")