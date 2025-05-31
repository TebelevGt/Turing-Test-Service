from pydantic import BaseModel

class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    is_bot_probability: float
