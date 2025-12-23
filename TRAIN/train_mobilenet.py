import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20  
LEARNING_RATE = 1e-4
DATA_DIR = 'dataset_final_ready' 

MODEL_NAME = 'tubris_mobilenet_clahe_final.h5'

def process_image_clahe(file_path):
    try:
        path_str = file_path.numpy().decode('utf-8')
    except AttributeError:
        path_str = str(file_path)

    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    
    if img is None: 
        return np.zeros((224,224,3), dtype=np.float32)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img)

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
    new_img = tf.keras.applications.mobilenet_v2.preprocess_input(new_img)
    
    return new_img

def tf_process_clahe(file_path, label):
    [image,] = tf.py_function(process_image_clahe, [file_path], [tf.float32])
    image.set_shape([224, 224, 3])
    return image, label

def create_dataset(subset):
    dir_path = os.path.join(DATA_DIR, subset)
    if not os.path.exists(dir_path):
        print(f"❌ ERROR: Folder tidak ditemukan: {dir_path}")
        return None

    normal_path = os.path.join(dir_path, 'Normal')
    tb_path = os.path.join(dir_path, 'Tuberculosis')
    
    normal_files = [os.path.join(normal_path, f) for f in os.listdir(normal_path)]
    tb_files = [os.path.join(tb_path, f) for f in os.listdir(tb_path)]
    
    all_files = normal_files + tb_files
    labels = [0]*len(normal_files) + [1]*len(tb_files)
    
    print(f"   {subset.upper()}: {len(all_files)} gambar ditemukan.")

    ds = tf.data.Dataset.from_tensor_slices((all_files, labels))
    ds = ds.shuffle(len(all_files))
    ds = ds.map(tf_process_clahe, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def build_mobilenet_clahe():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x) 
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    print(f"🚀 Memulai Training MobileNetV2 (CLAHE Version)...")
    train_ds = create_dataset('train')
    val_ds = create_dataset('test') 
    
    if train_ds is None or val_ds is None:
        print("❌ STOP: Dataset tidak lengkap. Cek folder 'dataset_final_ready'.")
        exit()
    
    model = build_mobilenet_clahe()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=2, factor=0.5, monitor='val_loss'),
        ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_loss')
    ]
  
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print(f"✅ Training Selesai!")

    print("💾 Menyimpan model final secara manual...")
    model.save(MODEL_NAME)
    print(f"✅ Model tersimpan di: {MODEL_NAME}")
    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss Curve')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.savefig('training_curve_mobilenet_clahe.png')
        print("📊 Grafik training disimpan.")
    except Exception as e:
        print(f"⚠️ Gagal save grafik: {e}")