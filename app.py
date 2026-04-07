from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier, Pool
import logging
import uvicorn
import os
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пути к артефактам (относительно текущего файла)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.cbm")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")

model = None
pipeline = None


class PredictionRequest(BaseModel):
    """Запрос на предсказание"""
    description: str = Field(..., min_length=1, max_length=5000, description="Текст объявления о ремонтных работах")


class PredictionResponse(BaseModel):
    """Ответ с результатом предсказания"""
    should_split: bool
    probability: float
    features_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan-контекст для загрузки/выгрузки ресурсов"""
    global model, pipeline

    logger.info("🚀 Загрузка артефактов модели...")

    try:
        # Проверка наличия файлов
        for path, name in [(MODEL_PATH, "model.cbm"), (VECTORIZER_PATH, "tfidf_vectorizer.joblib")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Не найден файл: {name} по пути {path}")

        # Загрузка модели CatBoost
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        logger.info("✅ Модель CatBoost загружена: %s", MODEL_PATH)

        # Загрузка пайплайна признаков с векторизатором
        from feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline(tfidf_vectorizer_path=VECTORIZER_PATH)
        logger.info("✅ Пайплайн признаков загружен: %s", VECTORIZER_PATH)

        logger.info("✨ Сервис готов к работе")
        yield

    except FileNotFoundError as exc:
        logger.error("❌ Критическая ошибка: файл артефакта не найден - %s", exc)
        raise RuntimeError("Не удалось загрузить артефакты модели") from exc
    except Exception as exc:
        logger.error("❌ Ошибка инициализации сервиса: %s", exc)
        raise RuntimeError("Не удалось инициализировать сервис") from exc
    finally:
        logger.info("🔄 Завершение работы сервиса")


app = FastAPI(
    title="Construction Split Prediction API",
    description="API для предсказания возможности разделения ремонтных работ на подкатегории",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Эндпоинт проверки работоспособности сервиса"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Выполняет предсказание: можно ли разделить описание на подкатегории.
    """
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Сервис не готов: модель или пайплайн не загружены")

    try:
        # 1. Предобработка: текст → 138 числовых признаков
        features_df = pipeline.transform([request.description])

        # 🔧 ДОБАВИТЬ: сырой текст для CatBoost text_features
        features_df['description_text'] = request.description

        # Логирование для отладки
        logger.debug("Сформировано признаков: %d, колонки: %s",
                     features_df.shape[1], list(features_df.columns[-3:]))

        # 2. Инференс через CatBoost
        # CatBoost автоматически обработает text_features при передаче DataFrame
        prediction = model.predict(features_df)
        probabilities = model.predict_proba(features_df)

        # 3. Формирование ответа
        return PredictionResponse(
            should_split=bool(prediction[0]),
            probability=float(probabilities[0][1]),
            features_count=features_df.shape[1] - 1  # Исключаем description_text из подсчёта
        )

    except ValueError as ve:
        logger.warning("Ошибка валидации входных данных: %s", ve)
        raise HTTPException(
            status_code=400,
            detail=f"Некорректные входные данные: {str(ve)}"
        ) from ve
    except Exception as exc:
        logger.error("Непредвиденная ошибка при предсказании: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервиса предсказаний"
        ) from exc


@app.get("/docs", include_in_schema=False)
async def redirect_docs():
    """Перенаправление на Swagger UI"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/redoc")


if __name__ == "__main__":
    logger.info("🔧 Запуск сервера на порту 8000...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENV") == "development",
        log_level="info"
    )