from flask import Flask, request, jsonify
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

print("STEP 1: Imports completed", flush=True)

app = Flask(__name__)

print("STEP 2: Flask started", flush=True)

CATEGORIES = ["Tools", "Electronics", "Books", "Sports", "Other"]

MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

print("STEP 3: Loading tokenizer", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("STEP 4: Tokenizer loaded", flush=True)
print("STEP 5: Loading ONNX model", flush=True)

model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
file_name="onnx/model_quint8_avx512.onnx"
)

print("STEP 6: ONNX model loaded", flush=True)
print("STEP 7: Creating classifier", flush=True)

classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer
)

print("STEP 8: Classifier ready", flush=True)


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()

    title = data.get("title", "")
    description = data.get("description", "")
    text = f"{title}. {description}".strip()

    if not text:
        return jsonify({"error": "Title or description is required"}), 400

    result = classifier(
        text,
        candidate_labels=CATEGORIES
    )

    return jsonify({
        "category": result["labels"][0],
        "score": result["scores"][0]
    })


@app.route("/")
def home():
    return jsonify({
        "message": "SmartShare AI API is running"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
