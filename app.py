from flask import Flask, request, render_template, url_for
import os
import numpy as np
import xgboost as xgb
import joblib
import json
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
from PIL import Image

# ===== Initialize Flask app =====
app = Flask(__name__)

# ===== Load models =====
cnn = load_model('models/cnn_feature_extractor.h5')
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('models/xgb_classifier.json')

# ===== Load labels =====
labels = joblib.load('models/labels.pkl')
idx2label = {v: k for k, v in labels.items()}

# ===== Label → Friendly names =====
fossil_map = {
    "Trilobite_Class": "Trilobite",
    "Belemnites_Class": "Belemnites",
    "Corals_Class": "Corals",
    "Crinoids_Class": "Crinoids",
    "Leaf Fossil_Class": "Leaf Fossil",
    "Ammonite_Class": "Ammonite",
    "Not_Fossil_Class": "Not a Fossil"  # <— NEW CLASS
}

# ===== Load fossil info & summary JSON =====
with open('fossil_info.json', 'r') as f:
    fossil_data = json.load(f)

with open('ai_summary.json', 'r') as f:
    summary_data = json.load(f)

# ===== Helper function for flexible fossil info lookup =====
def find_fossil_info(name):
    name_lower = name.lower().strip()
    for key, value in fossil_data.items():
        key_lower = key.lower().strip()
        if (
            key_lower == name_lower
            or key_lower in name_lower
            or name_lower in key_lower
            or key_lower.rstrip('s') == name_lower.rstrip('s')
        ):
            return value
    return {
        "class": "Unknown",
        "era": "Unknown",
        "diet": "Unknown",
        "geological_period": "Unknown",
        "habitat": "Unknown",
        "estimated_lifespan_years": "Unknown",
        "locations": [],
        "notes": "No additional info available."
    }

# ===== Flexible AI summary lookup =====
def get_ai_summary(name):
    name_lower = name.lower().strip()
    for key, value in summary_data.items():
        key_lower = key.lower().strip()
        if (
            key_lower == name_lower
            or key_lower in name_lower
            or name_lower in key_lower
            or key_lower.rstrip('s') == name_lower.rstrip('s')
        ):
            return value.get("summary", "No summary available.")
    return "No summary available."

# ===== Routes =====
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded!"

    file = request.files['image']
    filename = file.filename.lower()

    # ===== Step 1: Validate file type =====
    allowed_ext = {'jpg', 'jpeg', 'png'}
    if not ('.' in filename and filename.rsplit('.', 1)[1] in allowed_ext):
        return "Unsupported file format! Please upload JPG or PNG."

    # ===== Step 2: Create upload folder =====
    os.makedirs('static/uploads', exist_ok=True)
    file_path = os.path.join('static/uploads', filename)

    # ===== Step 3: Ensure readable image =====
    try:
        # Convert to RGB (fixes HEIC/WEBP/CMYK issues)
        image = Image.open(file.stream).convert('RGB')
        image.save(file_path, format='JPEG')
    except Exception as e:
        print("Image load error:", e)
        return "Error reading image file!"

    # ===== Step 4: Preprocess & predict =====
    try:
        img = preprocess_image(file_path)
        features = cnn.predict(img)
        probs = xgb_model.predict_proba(features)[0]
        pred_idx = np.argmax(probs)
        fossil_name = idx2label[pred_idx]
        confidence = probs[pred_idx] * 100
    except Exception as e:
        print("Prediction error:", e)
        return "Error during prediction!"

    # ===== Step 5: Check for 'Not a Fossil' condition =====
    fossil_name_friendly = fossil_map.get(fossil_name, fossil_name).strip()

    if fossil_name_friendly.lower() == "not a fossil" or confidence < 50:
        print("Detected as Non-Fossil (Confidence:", round(confidence, 2), "%)")
        return render_template(
            'not_fossil.html',
            image_path=file_path,
            confidence=round(confidence, 2)
        )

    # ===== Step 6: Fetch fossil info and AI summary =====
    fossil_info = find_fossil_info(fossil_name_friendly)
    ai_summary = get_ai_summary(fossil_name_friendly)

    # ===== Debug prints =====
    print("Uploaded file:", filename)
    print("Predicted fossil:", fossil_name_friendly)
    print("Confidence:", round(confidence, 2))
    print("Feature shape:", features.shape)
    print("Top 5 probs:", np.sort(probs)[::-1][:5])

    # ===== Step 7: Render result =====
    return render_template(
        'result.html',
        fossil=fossil_name_friendly,
        confidence=round(confidence, 2),
        image_path=file_path,
        fossil_info=fossil_info,
        ai_summary=ai_summary
    )

if __name__ == "__main__":
    app.run(debug=True)
