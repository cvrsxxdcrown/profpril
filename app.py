import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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


st.subheader("🤖 Машинное обучение — классификация вида цветка")

# Выбор признаков для обучения модели
features = st.multiselect("Выберите признаки для обучения модели", options=df.columns[:-1], default=list(df.columns[:-1]))

if len(features) < 1:
    st.warning("Пожалуйста, выберите хотя бы один признак для обучения модели.")
else:
    X = filtered_df[features]
    y = filtered_df["species"]

    # Разделение на train и test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Точность модели на тестовой выборке: **{acc:.2%}**")

    # Ввод данных для предсказания
    st.markdown("### Введите параметры для предсказания вида цветка")
    input_data = {}
    for feat in features:
        val = st.number_input(f"{feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))
        input_data[feat] = val

    if st.button("Сделать предсказание"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"Предсказанный вид цветка: **{prediction}**")
