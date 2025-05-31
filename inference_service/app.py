import logging
import os
import random

import mlflow.pyfunc
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from schemas import InferenceRequest, InferenceResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seed: int = 42
model = None

'''load_dotenv()
random.seed(seed)
np.random.seed(seed)

MLFLOW_PORT = os.environ.get("MLFLOW_PORT", "5000")
MLFLOW_TRACKING_URI = f"http://mlflow:{MLFLOW_PORT}"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
MODEL_URI = "models:/classifier@champion"'''

app = FastAPI(
    title="Inference Service",
    description="Сервис инференса для определения is_bot_probability",
)


@app.on_event("startup")
def load_model() -> None:
    global model
    try:
        logger.info("Начинается загрузка дообученной модели из локальной файловой системы")
        model =  mlflow.sklearn.load_model("models/")
        logger.info("Модель успешно загружена из локальной файловой системы")
    except Exception as e:
        logger.exception("Ошибка загрузки модели: %s", e)
        raise RuntimeError("Не удалось загрузить модель из локальной файловой системы.") from e


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest) -> InferenceResponse:
    """Эндпоинт инференса.
    Принимает текстовый запрос и возвращает вероятность того, что в диалоге присутствует бот.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Пустой текст не допускается.")

    try:
        raw_pred = model.predict_proba([text])
        print(raw_pred)
        is_bot_prob = float(raw_pred[:, 0])
        return InferenceResponse(is_bot_probability=is_bot_prob)
    except Exception as e:
        logger.exception("Ошибка во время инференса: %s", e)
        raise HTTPException(status_code=500, detail="Ошибка при обработке запроса.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, log_level="info")
