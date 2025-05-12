import subprocess

try:
    subprocess.check_call(["pip", "install", "--upgrade", "pip"])
    subprocess.check_call(["pip", "install", "--upgrade", "setuptools"])
    subprocess.check_call(["pip", "install", "--upgrade", "wheel"])
    subprocess.check_call(["pip", "install", "--upgrade", "numpy"])
except subprocess.CalledProcessError as e:
    print(f"Ошибка при установке зависимостей: {e}")
Ц
import streamlit as st
import pandas as pd

# Загрузка данных
st.title("Анализ данных Kaggle")
file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)
        st.write("Первые 5 строк данных:")
        st.write(df.head())

        # Контролы:
        st.sidebar.title("Параметры отображения")
        show_data = st.sidebar.checkbox("Показать все данные")
        column = st.sidebar.selectbox("Выберите колонку для анализа", df.columns)

        if show_data:
            st.write(df)

    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
else:
    st.info("Пожалуйста, загрузите CSV файл.")
