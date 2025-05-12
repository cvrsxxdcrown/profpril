import streamlit as st
import pandas as pd

# Загрузка данных
st.title("Анализ данных Kaggle")
file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Первые 5 строк данных:")
    st.write(df.head())

    # Контролы:
    st.sidebar.title("Параметры отображения")
    st.sidebar.checkbox("Показать все данные", key="show_data")
    st.sidebar.selectbox("Выберите колонку для анализа", df.columns)

    if st.sidebar.checkbox("Показать все данные"):
        st.write(df)
