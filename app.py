import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Конфигурация страницы
st.set_page_config(page_title="Iris Dataset Explorer", layout="centered")

# Заголовок
st.title("🌸 Исследование датасета Iris")

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv("iris.csv")

df = load_data()

st.subheader("📄 Первые строки данных")
st.dataframe(df.head())

# Контрол 1: Выбор вида цветка
species = st.multiselect("Выберите вид цветка", options=df["species"].unique(), default=df["species"].unique())

# Фильтрация данных
filtered_df = df[df["species"].isin(species)]

# Контрол 2: Слайдеры
sepal_length_range = st.slider("Диапазон длины чашелистика (sepal length)", 
                               float(df["sepal_length"].min()), float(df["sepal_length"].max()), 
                               (float(df["sepal_length"].min()), float(df["sepal_length"].max())))

# Контрол 3: Чекбокс
show_summary = st.checkbox("Показать статистическое резюме")

# Фильтрация по диапазону
filtered_df = filtered_df[
    (filtered_df["sepal_length"] >= sepal_length_range[0]) & 
    (filtered_df["sepal_length"] <= sepal_length_range[1])
]

# Контрол 4: Гистограмма
st.subheader("📊 Гистограмма длины лепестка (petal length)")
fig, ax = plt.subplots()
sns.histplot(filtered_df["petal_length"], kde=True, ax=ax)
st.pyplot(fig)

# Контрол 5: Выбор X и Y для графика
st.subheader("🔬 Диаграмма рассеяния")
x_axis = st.selectbox("Выберите X", df.columns[:-1], index=0)
y_axis = st.selectbox("Выберите Y", df.columns[:-1], index=1)

fig2, ax2 = plt.subplots()
sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue="species", ax=ax2)
st.pyplot(fig2)

# Показать описание
if show_summary:
    st.subheader("📈 Статистика")
    st.write(filtered_df.describe())
