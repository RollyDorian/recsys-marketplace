import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from search import ARTIFACT_DIR, ROOT, clean_text


def download_missing(paths):
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return

    import boto3

    credentials_path = Path(os.getenv('RECSYS_CREDENTIALS', ROOT.parent / '.local' / 'credentials.json'))
    if not credentials_path.is_file():
        raise SystemExit('Положите исходные Parquet в data/ или укажите RECSYS_CREDENTIALS.')

    credentials = json.loads(credentials_path.read_text(encoding='utf-8'))['student_s3']
    s3 = boto3.client(
        's3', endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=credentials['access_key'],
        aws_secret_access_key=credentials['secret_key'],
    )
    for path in missing:
        path.parent.mkdir(exist_ok=True)
        s3.download_file('s3-ds-source', path.name, str(path))


def main():
    raw_path = ROOT / 'data' / 'wb_products_raw_sample.parquet'
    pool_path = ROOT / 'data' / 'wb_products_dedup.parquet'
    download_missing([raw_path, pool_path])

    text_columns = ['imt_name', 'subj_name', 'description', 'brand_name']
    catalog = pd.read_parquet(
        raw_path, columns=['imt_id', 'nm_id', *text_columns],
        filters=[('subj_root_name', 'in', ['Одежда', 'Обувь'])],
        dtype_backend='pyarrow',
    )
    catalog['description_length'] = catalog['description'].fillna('').str.len()
    catalog = catalog.sort_values(
        ['description_length', 'nm_id'], ascending=[False, True], kind='stable'
    ).drop_duplicates('imt_id')
    for column in text_columns:
        catalog[column] = catalog[column].map(clean_text)
    has_text = catalog['imt_name'].ne('') | catalog['description'].ne('')
    catalog = catalog.loc[has_text].sort_values('imt_id').reset_index(drop=True)
    catalog['product_text'] = catalog[text_columns].apply(
        lambda row: ' . '.join(value for value in row if value), axis=1
    )

    pool_ids = pd.read_parquet(pool_path, columns=['imt_id'])['imt_id']
    in_pool = catalog['imt_id'].isin(pool_ids)
    products = catalog.loc[in_pool, ['imt_id', *text_columns, 'product_text']]
    if products.empty or set(products['imt_id']) != set(pool_ids):
        raise ValueError('Подготовленный каталог не совпадает с пулом оценки.')

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, dtype=np.float32)
    # idf считаем по всему каталогу одежды и обуви
    vectors = vectorizer.fit_transform(catalog['product_text'])[in_pool.to_numpy()]
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    products.to_parquet(ARTIFACT_DIR / 'catalog.parquet', index=False)
    save_npz(ARTIFACT_DIR / 'vectors.npz', vectors)
    joblib.dump(vectorizer, ARTIFACT_DIR / 'vectorizer.joblib')
    print(f'Сохранено товаров: {len(products)}. Папка: {ARTIFACT_DIR}')


if __name__ == '__main__':
    main()
