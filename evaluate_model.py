import os
import numpy as np
from tensorflow.keras.models import load_model
import xgboost as xgb
import joblib
from preprocess import preprocess_image

# Load models
cnn = load_model('models/cnn_feature_extractor.h5')
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('models/xgb_classifier.json')

# Load labels
labels = joblib.load('models/labels.pkl')
idx2label = {v: k for k, v in labels.items()}

# Test dataset path
test_path = 'test_data'

y_true = []
y_pred = []

# Iterate over classes
for class_name in os.listdir(test_path):
    class_folder = os.path.join(test_path, class_name)
    if not os.path.isdir(class_folder):
        continue
    
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        
        # Preprocess
        img = preprocess_image(img_path)
        features = cnn.predict(img)
        
        # Predict with XGBoost
        probs = xgb_model.predict_proba(features)[0]
        pred_idx = np.argmax(probs)
        pred_label = idx2label[pred_idx]
        
        y_true.append(class_name)
        y_pred.append(pred_label)

# Calculate accuracy
y_true = np.array(y_true)
y_pred = np.array(y_pred)
accuracy = np.sum(y_true == y_pred) / len(y_true)
print(f"Test accuracy: {accuracy * 100:.2f}%")
