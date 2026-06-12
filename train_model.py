import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xgboost as xgb
import joblib

# ========== CONFIG ==========
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_DIR = 'data/train'

# Step 1 - CNN feature extractor
cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten()
])

# Step 2 - Data loader
datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

# Step 3 - Extract features from images using CNN
features = []
labels = []

for i in range(len(train_gen)):
    x, y = train_gen[i]
    f = cnn.predict(x)
    features.append(f)
    labels.append(y)

features = np.vstack(features)
labels = np.hstack(labels)

# Step 4 - Train XGBoost on extracted features
xgb_model = xgb.XGBClassifier()
xgb_model.fit(features, labels)

# Step 5 - Save models
os.makedirs('models', exist_ok=True)
cnn.save('models/cnn_feature_extractor.h5')
xgb_model.save_model('models/xgb_classifier.json')
joblib.dump(train_gen.class_indices, 'models/labels.pkl')

print("✅ Model trained and saved in 'models/' folder!")
