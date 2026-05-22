import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import SeqIO
import re

@st.cache_data
def load_dataset():
    df = pd.read_csv("human_proteins_clean.csv", usecols=["ID", "Sequence", "Length"])
    return df.set_index("ID")

@st.cache_data
def analyze_protein(seq):
    if not seq:
        return {}, 0.0, 0.0, 0.0

    analysis = ProteinAnalysis(seq)
    aa_count = analysis.amino_acids_percent
    aa_count = {k: v * 100 for k, v in aa_count.items()}
    hydrophobicity = analysis.gravy()
    mw = analysis.molecular_weight()
    pi = analysis.isoelectric_point()
    return aa_count, hydrophobicity, mw, pi

def kyte_doolittle(seq, window=11):
    if not seq:
        return np.array([])
    
    kd_scale = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    scores = [kd_scale.get(aa, 0) for aa in seq]
    
    actual_window = min(window, len(scores))
    
    if actual_window == 0:
        return np.array([])
        
    windows = []
    for i in range(len(scores) - actual_window + 1):
        windows.append(np.mean(scores[i:i+actual_window]))
    return np.array(windows)

@st.cache_data
def find_similar_sequences(user_seq, database_df, num_results=5):
    cleaned_user_seq = "".join(aa for aa in user_seq.upper() if aa in "ACDEFGHIKLMNPQRSTVWY")
    if not cleaned_user_seq:
        return []

    results = []
    len_threshold_percent = 0.20
    len_threshold_abs = len(cleaned_user_seq) * len_threshold_percent
    
    candidate_df = database_df[
        (database_df["Length"] >= len(cleaned_user_seq) - len_threshold_abs) & 
        (database_df["Length"] <= len(cleaned_user_seq) + len_threshold_abs)
    ]
    
    if candidate_df.empty:
        return []

    for index, row in candidate_df.iterrows():
        raw_db_seq = str(row["Sequence"])
        cleaned_db_seq = "".join(aa for aa in raw_db_seq.upper() if aa in "ACDEFGHIKLMNPQRSTVWY")
        
        if not cleaned_db_seq:
            continue

        try:
            alignments = pairwise2.align.localxx(cleaned_user_seq, cleaned_db_seq, 1.0, -1.0, -5.0)
            
            if alignments:
                best_alignment = alignments[0]
                score = best_alignment.score
                
                num_identical = 0
                len_alignment = 0
                
                for i in range(len(best_alignment.seqA)):
                    if best_alignment.seqA[i] != '-' and best_alignment.seqB[i] != '-':
                        len_alignment += 1
                    if best_alignment.seqA[i] == best_alignment.seqB[i]:
                        num_identical += 1
                
                identity_percent = (num_identical / len_alignment * 100) if len_alignment > 0 else 0

                results.append({
                    "ID": index,
                    "База данных Длина": row["Length"],
                    "Пользовательская Длина": len(cleaned_user_seq),
                    "Счет выравнивания": score,
                    "Процент идентичности": identity_percent,
                    "Выравнивание": format_alignment(*best_alignment)
                })
        except ValueError:
            continue
        except TypeError as e:
            st.error(f"TypeError при выравнивании белка {index}: {e}. Последовательность: {cleaned_db_seq[:50]}...")
            continue

    results.sort(key=lambda x: x["Процент идентичности"], reverse=True)
    return results[:num_results]


def main():
    st.set_page_config(page_title="ProAn", layout="wide")
    st.title("🧬 ProAn — Анализ белков")
    st.write("--- Версия кода: 2023-11-20-4 ---")

    df = load_dataset()

    st.sidebar.header("Настройки")

    st.sidebar.subheader("Пользовательская последовательность")
    user_sequence_input = st.sidebar.text_area("Введите последовательность белка", height=150, help="Только заглавные буквы, A-Z")
    uploaded_file = st.sidebar.file_uploader("Или загрузите FASTA файл", type=["fasta", "txt"])

    custom_seq = ""
    custom_id = "Пользовательский белок" 

    if uploaded_file is not None:
        try:
            fasta_sequences = list(SeqIO.parse(uploaded_file, "fasta"))
            if fasta_sequences:
                temp_seq = str(fasta_sequences[0].seq).strip().upper()
                if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in temp_seq) and len(temp_seq) > 0:
                    custom_seq = temp_seq
                    st.sidebar.success("Последовательность из FASTA загружена.")
                else:
                    st.sidebar.error("Последовательность из FASTA содержит недопустимые символы или пуста.")
            else:
                st.sidebar.error("В FASTA файле не найдено ни одной последовательности.")
        except Exception as e:
            st.sidebar.error(f"Ошибка при чтении FASTA файла: {e}")
    elif user_sequence_input:
        temp_seq = user_sequence_input.strip().upper()
        if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in temp_seq) and len(temp_seq) > 0:
            custom_seq = temp_seq
            st.sidebar.success("Последовательность введена вручную.")
        else:
            st.sidebar.error("Введенная последовательность содержит недопустимые символы или пуста.")
    
    st.sidebar.markdown("---")

    st.sidebar.subheader("Фильтры базы данных")

    min_len_filter, max_len_filter = st.sidebar.slider(
        "Фильтр по длине белка",
        int(df["Length"].min()),
        int(df["Length"].max()),
        (int(df["Length"].min()), int(df["Length"].max()))
    )
    
    search_term = st.sidebar.text_input("Поиск по ID:", "")

    filtered_df = df[
        (df["Length"] >= min_len_filter) & (df["Length"] <= max_len_filter)
    ]
    if search_term:
        filtered_df = filtered_df[
            filtered_df.index.str.contains(search_term, case=False)
        ]

    sort_by_options = {
        "Длина (убыв.)": ("Length", False),
        "Длина (возр.)": ("Length", True),
        "ID (убыв.)": ("ID", False),
        "ID (возр.)": ("ID", True)
    }
    sort_option_label = st.sidebar.selectbox("Сортировать по:", list(sort_by_options.keys()))
    sort_column, sort_ascending = sort_by_options[sort_option_label]

    if sort_column == "ID":
        filtered_df = filtered_df.sort_index(ascending=sort_ascending)
    else:
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending)

    if filtered_df.empty:
        st.sidebar.warning("Нет белков, соответствующих фильтрам. Измените фильтры.")
        selected_protein_id = None
        if not custom_seq:
            st.info("Введите пользовательскую последовательность или измените фильтры, чтобы увидеть белки из базы данных.")
    else:
        selected_protein_id = st.sidebar.selectbox(
            "Выберите белок из базы данных:", 
            filtered_df.index.tolist(), 
            key="db_protein_select"
        )
    
    if custom_seq:
        st.header(f"Анализ: Пользовательский белок")
        
        custom_aa_comp, custom_gravy, custom_mw, custom_pi = analyze_protein(custom_seq)
        
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric("Длина", len(custom_seq))
            st.metric("GRAVY", f"{custom_gravy:.2f}")
        with col_d:
            st.metric("Молекулярная масса", f"{custom_mw:.0f} Da")
            st.metric("pI", f"{custom_pi:.2f}")
            
        st.subheader("Визуализация для пользовательской последовательности")
        col_e, col_f = st.columns(2)
        with col_e:
            custom_aa_df = pd.DataFrame(list(custom_aa_comp.items()), columns=["AA", "Percent"])
            fig_pie_custom = px.pie(custom_aa_df, values="Percent", names="AA", title="Состав аминокислот (%)")
            st.plotly_chart(fig_pie_custom, use_container_width=True)
        
        with col_f:
            custom_kd_profile = kyte_doolittle(custom_seq)
            fig_kd_custom = go.Figure()
            fig_kd_custom.add_trace(go.Scatter(y=custom_kd_profile, mode='lines', name='Kyte-Doolittle'))
            fig_kd_custom.add_hline(y=0, line_dash="dash", line_color="red")
            fig_kd_custom.update_layout(title="Профиль гидрофобности", xaxis_title="Позиция", yaxis_title="Гидрофобность")
            st.plotly_chart(fig_kd_custom, use_container_width=True)

        st.subheader("Поиск похожих белков в базе данных")
        st.write("Используется попарное выравнивание (Smith-Waterman) для оценки сходства.")
        
        num_results_to_show = st.slider("Показать топ N результатов:", 1, 10, 5)
        
        if st.button("Найти совпадения в базе"):
            if not custom_seq:
                st.warning("Пожалуйста, введите или загрузите пользовательскую последовательность для поиска.")
            else:
                with st.spinner('Выполняется поиск совпадений... Это может занять некоторое время для длинных последовательностей.'):
                    similar_proteins = find_similar_sequences(custom_seq, df, num_results=num_results_to_show)
                
                if similar_proteins:
                    st.write(f"Найдено {len(similar_proteins)} похожих белков:")
                    for i, prot in enumerate(similar_proteins):
                        st.markdown(f"**{i+1}. ID: {prot['ID']}**")
                        st.write(f"   Длина в базе: {prot['База данных Длина']} | Ваша длина: {prot['Пользовательская Длина']}")
                        st.write(f"   Процент идентичности: {prot['Процент идентичности']:.2f}%")
                        st.write(f"   Счет выравнивания: {prot['Счет выравнивания']:.2f}")
                        with st.expander(f"Показать выравнивание для {prot['ID']}"):
                            st.code(prot['Выравнивание'], language="text")
                        st.markdown("---")
                else:
                    st.write("Похожих белков не найдено.")
        
        st.markdown("---")

    if selected_protein_id:
        st.header(f"Анализ: {selected_protein_id}")
        protein_info = df.loc[selected_protein_id]
        seq = protein_info["Sequence"]
        
        aa_comp, gravy, mw, pi = analyze_protein(seq)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.metric("Длина", protein_info["Length"])
            st.metric("Гидрофобность (GRAVY)", f"{gravy:.2f}")
        with col2:
            st.metric("Молекулярная масса", f"{mw:.0f} Da")
            st.metric("pI", f"{pi:.2f}")
        
        col_pie, col_kd = st.columns(2)
        
        with col_pie:
            aa_df = pd.DataFrame(list(aa_comp.items()), columns=["AA", "Percent"])
            fig_pie = px.pie(aa_df, values="Percent", names="AA", title="Состав аминокислот (%)")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_kd:
            kd_profile = kyte_doolittle(seq)
            fig_kd = go.Figure()
            fig_kd.add_trace(go.Scatter(y=kd_profile, mode='lines', name='Kyte-Doolittle'))
            fig_kd.add_hline(y=0, line_dash="dash", line_color="red")
            fig_kd.update_layout(title="Профиль гидрофобности", xaxis_title="Позиция", yaxis_title="Гидрофобность")
            st.plotly_chart(fig_kd, use_container_width=True)
        
        st.header("📋 Сводная таблица")
        summary = {
            "ID": selected_protein_id,
            "Длина": protein_info["Length"],
            "GRAVY": f"{gravy:.2f}",
            "Масса (Da)": f"{mw:.0f}",
            "pI": f"{pi:.2f}"
        }
            
        st.table(pd.DataFrame([summary]))

if __name__ == "__main__":
    main()