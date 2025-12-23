import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


TRAIN_DIR = 'dataset_final_ready/train'
VALIDATION_DIR = 'dataset_final_ready/validation'
TEST_DIR = 'dataset_final_ready/test'

IMG_SIZE = (224, 224) 
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32 
EPOCHS = 20 

print("Memuat data...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary'
)

class_names = train_dataset.class_names
print(f"Kelas terdeteksi: {class_names}")

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

total_train_samples = 2449 + 489
weight_for_0 = (1 / 2449) * (total_train_samples / 2.0)
weight_for_1 = (1 / 489) * (total_train_samples / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"Bobot Kelas 0 (Normal): {weight_for_0:.2f}")
print(f"Bobot Kelas 1 (Tuberculosis): {weight_for_1:.2f}")
print("Membangun model EfficientNetB0...")

base_model = EfficientNetB0(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

inputs = keras.Input(shape=IMG_SHAPE)

x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = data_augmentation(x, training=True) 
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n--- Mulai Training (Frozen) ---")

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    class_weight=class_weight
)


print("\n--- Mulai Fine-Tuning (Unfrozen) ---")

base_model.trainable = True
fine_tune_at = 150 

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

TOTAL_EPOCHS = EPOCHS + 10 

history_fine_tune = model.fit(
    train_dataset,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
    class_weight=class_weight
)

print("--- Fine-Tuning Selesai ---")

acc = history.history['accuracy'] + history_fine_tune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
loss = history.history['loss'] + history_fine_tune.history['loss']
val_loss = history.history['val_loss'] + history_fine_tune.history['val_loss']

print("\n--- Evaluasi Test Set ---")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\n--- Laporan Klasifikasi ---")
y_true = []
for images, labels in test_dataset.unbatch():
    y_true.append(labels.numpy())
y_true = np.array(y_true)

y_pred_probs = model.predict(test_dataset)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([EPOCHS-1, EPOCHS-1], plt.ylim(), label='Mulai Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([EPOCHS-1, EPOCHS-1], plt.ylim(), label='Mulai Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model_filename = "tubris_effnet_clahe_final.h5" 
model.save(model_filename)
print(f"\nModel telah disimpan sebagai {model_filename}")