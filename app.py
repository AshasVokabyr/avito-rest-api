from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostClassifier
from feature_pipeline import FeaturePipeline
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Construction Split Prediction API", version="1.0.0")

# Пути к артефактам
MODEL_PATH = "model.cbm"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

model = None
pipeline = None

class PredictionRequest(BaseModel):
    description: str

@app.on_event("startup")
async def load_artifacts():
    global model, pipeline
    try:
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        
        pipeline = FeaturePipeline(tfidf_vectorizer_path=VECTORIZER_PATH)
        
        logger.info("Модель и пайплайн признаков успешно загружены")
    except Exception as exc:
        logger.error("Ошибка загрузки артефактов: %s", exc)
        raise RuntimeError("Не удалось инициализировать сервис") from exc

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    
    try:
        # 1. Предобработка данных
        features_df = pipeline.transform([request.description])
        
        # 2. Инференс
        prediction = model.predict(features_df)
        probabilities = model.predict_proba(features_df)
        
        return {
            "should_split": bool(prediction[0]),
            "probability": float(probabilities[0][1]),
            "features_count": features_df.shape[1]
        }
    except Exception as exc:
        logger.error("Ошибка при предсказании: %s", exc)
        raise HTTPException(status_code=500, detail="Ошибка выполнения предсказания")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)