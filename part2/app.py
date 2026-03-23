import logging
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]

app = FastAPI(title="rubert-mini-frida ONNX")

logger.info("Loading ONNX model...")
model = ORTModelForFeatureExtraction.from_pretrained("onnx_model")
tokenizer = AutoTokenizer.from_pretrained("onnx_model")
logger.info("ONNX model loaded!")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask[..., None]
    return (token_embeddings * input_mask_expanded).sum(axis=1) / input_mask_expanded.sum(axis=1)

@app.post("/embed", response_model=EmbedResponse)
async def get_embedding(request: EmbedRequest):
    inputs = tokenizer(
        request.texts,
        padding=True,
        truncation=True,
        return_tensors="np"
    )

    outputs = model(**inputs)

    embeddings = mean_pooling(outputs, inputs["attention_mask"])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return EmbedResponse(embeddings=embeddings.tolist())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)