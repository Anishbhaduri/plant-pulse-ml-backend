from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # allow frontend requests

# Load model
model = tf.keras.models.load_model("model.h5")

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___healthy"
]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file).resize((224, 224))
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)

    preds = model.predict(img)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)

    return jsonify({
        "class_index": class_index,
        "class_name": CLASS_NAMES[class_index],
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001)
