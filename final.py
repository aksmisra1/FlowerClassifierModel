from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Set image folder
data_dir = 'flowers'

# Basic image and batch size settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Setup image preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Training data
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

# Validation data
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# Show class names
print("Classes:", train_gen.class_indices)

# Show shape of 1 batch
images, labels = next(train_gen)
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# Load pretrained MobileNetV2 without top classification layer
num_classes = len(train_gen.class_indices)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model for now

# Build new classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess image
img_path = 'test_flower.jpeg'  # <-- replace with your file name
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
class_labels = list(train_gen.class_indices.keys())

print(f"Predicted flower type: {class_labels[predicted_class]}")
