import logging
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]

app = FastAPI(title="rubert-mini-frida baseline")

logger.info("Loading model...")
model = SentenceTransformer('sergeyzh/rubert-mini-frida', device='cpu')
logger.info("Model loaded!")

@app.post("/embed", response_model=EmbedResponse)
async def get_embedding(request: EmbedRequest):
    embeddings = model.encode(
        request.texts,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()
    return EmbedResponse(embeddings=embeddings)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)