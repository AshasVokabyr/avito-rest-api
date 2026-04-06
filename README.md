# Avito REST API

API для предсказания разделения работ на основе описания объявления на Avito. Использует модель CatBoost для классификации.

## Компоненты

| Компонент       | Технология                          | Причина                                      |
|-----------------|-------------------------------------|----------------------------------------------|
| Фреймворк       | FastAPI                             | Простота, автодокументация, валидация        |
| Сервер          | Uvicorn                             | ASGI сервер для FastAPI                      |
| Библиотека ML   | CatBoost                            | Поддержка `.cbm` формата                     |
| Данные          | JSON                                | Универсальный формат                         |
| Обработка текста| Scikit-learn (TfidfVectorizer)     | Векторизация текста                          |
| Данные          | Pandas                              | Работа с DataFrame                           |

## Структура проекта

```
avito-rest-api/
├── app.py                    # Основное приложение FastAPI
├── feature_pipeline.py       # Пайплайн обработки признаков
├── model.cbm                 # Файл обученной модели CatBoost
├── tfidf_vectorizer.joblib   # Векторизатор TF-IDF
├── requirements.txt          # Зависимости Python
├── README.md                 # Документация
├── venv/                     # Виртуальное окружение (не в репозитории)
└── .gitignore               # Игнорируемые файлы
```

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone <repository-url>
   cd avito-rest-api
   ```

2. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # На Windows: venv\Scripts\activate
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Запуск

Запустите сервер:
```bash
python app.py
```

Или с uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

API будет доступно по адресу: http://localhost:8000

Документация API: http://localhost:8000/docs

## API

### POST /predict

Предсказывает, нужно ли разделять работу на основе описания.

**Запрос:**
```json
{
  "description": "Текст описания объявления"
}
```

**Ответ:**
```json
{
  "should_split": true,
  "probability": 0.85,
  "features_count": 50
}
```

- `should_split`: boolean - нужно ли разделять работу
- `probability`: float - вероятность разделения
- `features_count`: int - количество извлеченных признаков

## Разработка

Для разработки установите дополнительные зависимости (если нужны) и используйте reload:
```bash
uvicorn app:app --reload
```