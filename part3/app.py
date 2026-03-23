import time
import queue
import threading
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from concurrent.futures import Future

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import uvicorn


MODEL_PATH = "../part2/onnx_model" 
BATCH_WAIT_TIME = 0.005
MAX_BATCH_SIZE = 32

request_queue = queue.Queue()
app = FastAPI(title="rubert-mini-frida ONNX dynamic batching")

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

print("Loading ONNX model...")
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Model loaded!")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask[..., None]
    return (token_embeddings * input_mask_expanded).sum(axis=1) / input_mask_expanded.sum(axis=1)

def batch_worker():
    while True:
        batch_texts = []
        futures = []
        start_time = time.time()

        while time.time() - start_time < BATCH_WAIT_TIME and len(batch_texts) < MAX_BATCH_SIZE:
            try:
                req_texts, fut = request_queue.get_nowait()
                batch_texts.extend(req_texts)
                futures.append((fut, len(req_texts)))
            except queue.Empty:
                time.sleep(0.001)

        if batch_texts:
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="np")
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs, inputs["attention_mask"])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = (embeddings / norms).tolist()

            idx = 0
            for fut, n_texts in futures:
                fut.set_result(embeddings[idx:idx+n_texts])
                idx += n_texts

threading.Thread(target=batch_worker, daemon=True).start()

@app.post("/embed", response_model=EmbedResponse)
def get_embedding(request: EmbedRequest):
    fut = Future()
    request_queue.put((request.texts, fut))
    embeddings = fut.result()
    return EmbedResponse(embeddings=embeddings)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)