import cv2
import numpy as np
from PIL import Image

IMG_SIZE = (128, 128)

def preprocess_image(image_path):
    # Try reading with OpenCV
    img = cv2.imread(image_path)
    
    # If OpenCV fails (img is None), use PIL instead
    if img is None:
        with Image.open(image_path) as pil_img:
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
    
    # Convert to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
