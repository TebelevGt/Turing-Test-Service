
import logging
import redis
import uuid
import os
import requests
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from data.schemas import IncomingMessage, Prediction
import hashlib
import time
# Инициализация логгера
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Настройка фастапи
FASTAPI_PORT = "8001"
app = FastAPI(
    title="Inference Service",
    description="Сервис для классификации диалогов на FastApi",
)

# Настройка редиса для кэшировани
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def make_cache_key(text: str) -> str:
    """Формирование ключа для кэша с помощью SHA-256, учитывая выбранный сервис.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """Эндпоинт для получения вероятности того,.

    что в диалоге участвует бот.

    Возвращаем объект `Prediction`.
    """
    cache_key = make_cache_key(msg.text)
    cached_prob = redis_client.get(cache_key)
    if cached_prob is not None:
        logger.info("Cache hit for key: %s", cache_key)
        return Prediction(
            id=uuid.uuid4(),
            message_id=msg.id,
            dialog_id=msg.dialog_id,
            participant_index=msg.participant_index,
            is_bot_probability=float(cached_prob),
        )




    url = f"http://inference-service:{FASTAPI_PORT}/predict"
    body = {"text": msg.text}

    try:
        resp = requests.post(url, json=body, timeout=3.0)
        resp.raise_for_status()
    except Exception as e:
        logger.exception("Ошибка при запросе к inference_service: %s", e)
        raise HTTPException(status_code=500, detail="Ошибка при запросе к inference_service")

    inference_response = resp.json()
    is_bot_probability = float(inference_response.get("is_bot_probability", 0.0))

    redis_client.set(cache_key, is_bot_probability)
    prediction_id = uuid.uuid4()
    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
