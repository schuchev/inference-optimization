from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_NAME = "sergeyzh/rubert-mini-frida"
SAVE_PATH = "onnx_model"

model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_NAME,
    export=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("Модель успешно сконвертирована в ONNX!")