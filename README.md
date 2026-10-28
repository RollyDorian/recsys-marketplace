# Семантический поиск товаров

Поиск товаров одежды и обуви по текстовому запросу. Проведены EDA и очистка данных, реализованы TF-IDF и семантический поиск на модели paraphrase-multilingual-MiniLM-L12-v2.

Решение находится в semantic_search_project.ipynb, выводы и ответы на вопросы QA - в semantic_search_project_workbook.md.

## Данные

В папке data/ используются четыре файла:

- wb_products_raw_sample.parquet
- wb_products_dedup.parquet
- queries_synthetic_train.parquet
- queries_synthetic_test.parquet

Если файлов нет, ноутбук скачивает их из s3-ds-source. Для этого нужно указать в RECSYS_CREDENTIALS путь к JSON с ключами student_s3.access_key и student_s3.secret_key.

## Запуск

Python 3.12, CPU. Установить зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Открыть semantic_search_project.ipynb с ядром из .venv и выполнить ячейки недель 1-2 по порядку. Модель скачивается при первом запуске, эмбеддинги сохраняются в artifacts/.

## Результаты

Оценка проведена на 700 train-запросах и общем пуле из 4 256 карточек. Бинарный порог релевантности - 2, значения K - 1, 3, 5, 10.

| Метод | NDCG@10 | Recall@10 | MRR |
|---|---:|---:|---:|
| TF-IDF | 0.4315 | 0.2869 | 0.1769 |
| Семантический поиск | 0.1512 | 0.1378 | 0.1233 |

TF-IDF показывает лучшее качество. В рассмотренных примерах семантический поиск возвращает товары других категорий. Подробное сравнение и примеры выдачи приведены в ноутбуке.
