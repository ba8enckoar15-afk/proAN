import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re

@st.cache_data
def load_dataset():
    df = pd.read_csv("human_proteins_clean.csv")
    return df.set_index("ID")

@st.cache_data
def analyze_protein(seq):
    analysis = ProteinAnalysis(seq)
    aa_count = analysis.amino_acids_percent
    aa_count = {k: v*100 for k, v in aa_count.items()}
    hydrophobicity = analysis.gravy()
    mw = analysis.molecular_weight()
    pi = analysis.isoelectric_point()
    return aa_count, hydrophobicity, mw, pi

def kyte_doolittle(seq, window=11):
    kd_scale = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    scores = [kd_scale.get(aa, 0) for aa in seq]
    windows = []
    for i in range(len(scores) - window + 1):
        windows.append(np.mean(scores[i:i+window]))
    return np.array(windows)

def main():
    st.set_page_config(page_title="ProAn", layout="wide")
    st.title("🧬 ProAn — Анализ белков")
    
    df = load_dataset()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("🔍 Поиск")
        protein_id = st.selectbox("Выберите белок:", df.index.tolist()[:1000])
        seq = df.loc[protein_id, "Sequence"]
        length = df.loc[protein_id, "Length"]
        
        st.metric("Длина", length)
    
    with col2:
        aa_comp, gravy, mw, pi = analyze_protein(seq)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Гидрофобность (GRAVY)", f"{gravy:.2f}")
        with col_b:
            st.metric("Молекулярная масса", f"{mw:.0f} Da")
    
    col1, col2 = st.columns(2)
    
    with col1:
        aa_df = pd.DataFrame(list(aa_comp.items()), columns=["AA", "Percent"])
        fig_pie = px.pie(aa_df, values="Percent", names="AA", title="Состав аминокислот (%)")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        kd_profile = kyte_doolittle(seq)
        fig_kd = go.Figure()
        fig_kd.add_trace(go.Scatter(y=kd_profile, mode='lines', name='Kyte-Doolittle'))
        fig_kd.add_hline(y=0, line_dash="dash", line_color="red")
        fig_kd.update_layout(title="Профиль гидрофобности", xaxis_title="Позиция", yaxis_title="Гидрофобность")
        st.plotly_chart(fig_kd, use_container_width=True)
    
    st.header("📋 Сводная таблица")
    summary = {
        "Длина": length,
        "GRAVY": gravy,
        "Масса (Da)": mw,
        "pI": pi
    }
    st.table(pd.DataFrame([summary]))

if __name__ == "__main__":
    main()
