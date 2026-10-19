from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)
HF_TOKEN = os.environ.get("HF_TOKEN")

HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

CATEGORIES = ["Tools", "Electronics", "Books", "Sports", "Other"]
@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()

    title = data.get("title", "")
    description = data.get("description", "")
    text = f"{title}. {description}".strip()

    if not text:
        return jsonify({"error": "Title or description is required"}), 400

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": CATEGORIES
        }
    }

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return jsonify({
            "error": "AI classification failed",
            "details": response.text
        }), response.status_code

    result = response.json()

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
