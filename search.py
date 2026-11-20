import html
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / 'artifacts' / 'app'


def clean_text(value):
    if pd.isna(value):
        return ''
    value = re.sub(r'<[^>]+>', ' ', html.unescape(str(value)))
    value = re.sub(r'https?://\S+|www\.\S+', ' ', value)
    value = re.sub(r'[^\w\s.,%+/-]', ' ', value, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', value).strip()


def search(query, catalog, vectorizer, vectors, top_k=10):
    query_vector = vectorizer.transform([clean_text(query)])
    scores = (vectors @ query_vector.T).toarray().ravel()
    positions = np.argsort(-scores, kind='stable')[:top_k]
    positions = positions[scores[positions] > 0]
    return catalog.iloc[positions].assign(score=scores[positions])
