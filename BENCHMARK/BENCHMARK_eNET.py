import os
import time
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet

MODEL_PATH = 'tubris_effnet_clahe_final.h5'
TEST_DIR = r'dataset_final_ready/test'
IMG_SIZE = (224, 224)
MODEL_NAME = "EfficientNetB0 (CLAHE)"
TRAINING_TIME = "55 menit"

def get_processed_image(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return None
    
    img_blurred = cv2.GaussianBlur(img_gray, (5,5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_blurred)
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
    
    old_size = img_rgb.shape[:2]
    ratio = float(IMG_SIZE[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img_resized = cv2.resize(img_rgb, (new_size[1], new_size[0]))
    delta_w = IMG_SIZE[1] - new_size[1]
    delta_h = IMG_SIZE[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    new_img = new_img.astype(np.float32)
    return efficientnet.preprocess_input(new_img)

print(f"🚀 MENJALANKAN BENCHMARK: {MODEL_NAME}")

if not os.path.exists(MODEL_PATH):
    print("❌ File model tidak ditemukan!")
    exit()

try:
    custom_objects = {'TrueDivide': tf.math.truediv}
    model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
except Exception as e:
    print(f"❌ Error Load: {e}")
    exit()

y_true = []
y_pred = []
classes = {'Normal': 0, 'Tuberculosis': 1}

for cls_name, cls_idx in classes.items():
    cls_folder = os.path.join(TEST_DIR, cls_name)
    if not os.path.exists(cls_folder): continue
    for fname in os.listdir(cls_folder):
        fpath = os.path.join(cls_folder, fname)
        img = get_processed_image(fpath)
        if img is not None:
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img, verbose=0)[0][0]
            y_true.append(cls_idx)
            y_pred.append(1 if pred > 0.5 else 0)

print("\n" + "="*50)
print(f"HASIL AKHIR: {MODEL_NAME}")
print("="*50)
print(f"Accuracy      : {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"Precision     : {precision_score(y_true, y_pred, zero_division=0)*100:.2f}%")
print(f"Sensitivity   : {recall_score(y_true, y_pred, zero_division=0)*100:.2f}%")
print(f"Training Time : {TRAINING_TIME}")
print("="*50)