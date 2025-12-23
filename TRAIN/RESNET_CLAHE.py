import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50V2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_DIR = 'dataset_final_ready/train'
VALIDATION_DIR = 'dataset_final_ready/validation'
TEST_DIR = 'dataset_final_ready/test'

IMG_SIZE = (224, 224) 
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32 
EPOCHS = 25 

print("Memuat dataset CLAHE...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='binary')
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='binary')
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='binary')

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

total = 3500 + 700
w0 = (1 / 3500) * (total / 2.0)
w1 = (1 / 700) * (total / 2.0)
class_weight = {0: w0, 1: w1}

print("Membangun ResNet50V2...")
base_model = ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False 

inputs = keras.Input(shape=IMG_SHAPE)
x = tf.keras.applications.resnet_v2.preprocess_input(inputs) 
x = data_augmentation(x, training=True)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Training Awal ---")
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, class_weight=class_weight)

print("\n--- Fine Tuning ---")
base_model.trainable = True
for layer in base_model.layers[:100]: layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy', metrics=['accuracy'])

total_epochs = EPOCHS + 10
history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset, class_weight=class_weight)

model_filename = "tubris_resnet_clahe_final.h5"
model.save(model_filename)
print(f"\n✅ Model Disimpan: {model_filename}")