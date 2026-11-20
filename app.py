import joblib
import pandas as pd
from scipy.sparse import load_npz
import streamlit as st

from search import ARTIFACT_DIR, search


@st.cache_resource
def load_index():
    vectorizer = joblib.load(ARTIFACT_DIR / 'vectorizer.joblib')
    return vectorizer, load_npz(ARTIFACT_DIR / 'vectors.npz')


@st.cache_data
def load_catalog():
    return pd.read_parquet(ARTIFACT_DIR / 'catalog.parquet')


st.set_page_config(page_title='Поиск товаров')
st.title('Поиск одежды и обуви')

try:
    vectorizer, vectors = load_index()
    catalog = load_catalog()
except FileNotFoundError:
    st.error('Сначала подготовьте файлы поиска по инструкции в README.')
    st.stop()

if vectors.shape != (len(catalog), len(vectorizer.vocabulary_)):
    st.error('Файлы поиска не совпадают. Подготовьте их заново по README.')
    st.stop()

with st.form('search'):
    query = st.text_input('Запрос', placeholder='Например, кожаные мокасины')
    submitted = st.form_submit_button('Найти')

if submitted:
    if not query.strip():
        st.info('Введите название товара, бренд или описание.')
    else:
        results = search(query, catalog, vectorizer, vectors)
        if results.empty:
            st.info('Совпадений нет. Попробуйте другой запрос.')
        for item in results.itertuples(index=False):
            st.divider()
            st.subheader(item.imt_name or item.subj_name or 'Товар без названия')
            brand = ' · ' + item.brand_name if item.brand_name else ''
            st.caption(f'{item.subj_name}{brand} · ID {item.imt_id} · Сходство {item.score:.3f}')
            if item.description:
                description = item.description[:220]
                st.text(description + ('...' if len(item.description) > 220 else ''))
