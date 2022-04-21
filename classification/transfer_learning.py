import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

image_dir = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "images")
batch_size = 32
img_size = (900, 900)


# Lade Training und Validation Bilder aus Ordner
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size)


# Zeige die ersten 9 Bilder des Datensatzes
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()


# Data Augmentation durch rotieren und spiegeln
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    # tf.keras.layers.experimental.preprocessing.RandomRotation(45),
])

# Zeige ein Bild in 9 verschiedenen Versionen
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

plt.show()


# Verwende MobileNet OHNE top layers
base_model = MobileNet(weights='imagenet', include_top=False)

# Einzelne neue Layer zum anpassen an eigene Daten hinzufügen
inputs = tf.keras.Input(shape=(900, 900, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
outputs = Dense(1, activation='softmax')(x)


# MobileNet und prediction Layer zusammenfügen
model = Model(inputs=inputs, outputs=outputs)


# Nur neue Layer sollen trainierbar sein
for layer in model.layers[:-3]:
    layer.trainable = False
for layer in model.layers[-3:]:
    layer.trainable = True

print(model.summary())


# Nutze preprocess Funktion von MobileNet um eigene Bilder vorzubereiten
#train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# train_generator = train_data_gen.flow_from_directory('./data_for_transfer_learning/',
#                                                    target_size=(224, 224),
#                                                    color_mode='rgb',
#                                                    batch_size=32,
#                                                    class_mode='categorical',
#                                                    shuffle=True)


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=5)


acc = history.history['accuracy']
loss = history.history['loss']

plt.figure(figsize=(8, 8))
plt.plot(acc, label='Training Accuracy')
plt.plot(loss, label='Training Loss')
plt.legend(loc='lower left')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training Accuracy and Loss')
plt.show()


# model.predict(new_image)
