from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

CATEGORIES = ["Tools", "Electronics", "Books", "Sports", "Other"]
MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()

    title = data.get("title", "")
    description = data.get("description", "")
    text = f"{title}. {description}".strip()

    if not text:
        return jsonify({"error": "Title or description is required"}), 400

    from transformers import pipeline

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME
    )

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
