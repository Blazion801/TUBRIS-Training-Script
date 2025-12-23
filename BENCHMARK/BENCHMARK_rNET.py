import os
import time
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

MODEL_PATH = 'tubris_resnet_clahe_final.h5' 
TEST_DIR = r'dataset_final_ready/test'
IMG_SIZE = (224, 224)
TRAINING_TIME = "1 jam 20 menit" 

def build_resnet():
    base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:100]: layer.trainable = False 
    
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs) 
    x = data_augmentation(x, training=False)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) 
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

def preprocess_clahe_consistent(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
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
    return np.expand_dims(new_img, axis=0)

print(f"🚀 MENJALANKAN BENCHMARK: ResNet50V2 (CLAHE)")

if not os.path.exists(MODEL_PATH):
    print("❌ File model tidak ditemukan!")
    exit()

try:
    model = build_resnet()
    model.load_weights(MODEL_PATH)
except Exception as e:
    print(f"❌ Error Load: {e}")
    exit()

y_true = []
y_pred = []
classes = {'Normal': 0, 'Tuberculosis': 1}

print("🔄 Sedang memproses data uji...")

for cls_name, cls_idx in classes.items():
    cls_folder = os.path.join(TEST_DIR, cls_name)
    if not os.path.exists(cls_folder): continue
    
    files = os.listdir(cls_folder)
    for fname in files:
        fpath = os.path.join(cls_folder, fname)
        input_data = preprocess_clahe_consistent(fpath)
        if input_data is not None:
            pred = model.predict(input_data, verbose=0)[0][0]
            y_true.append(cls_idx)
            y_pred.append(1 if pred > 0.5 else 0)

print("\n" + "="*50)
print(f"HASIL AKHIR: ResNet50V2 (CLAHE)")
print("="*50)
print(f"Accuracy      : {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"Precision     : {precision_score(y_true, y_pred, zero_division=0)*100:.2f}%")
print(f"Sensitivity   : {recall_score(y_true, y_pred, zero_division=0)*100:.2f}%")
print(f"Training Time : {TRAINING_TIME}")
print("="*50)