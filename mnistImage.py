#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import asarray
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
from PIL import Image


#Ingest Data
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
print(test_images[1].shape)
print(type(test_labels[0]))

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

num_filters = 32
filter_size = 3
pool_size = 2

# Build the model.
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1), activation="relu"),
  MaxPooling2D(pool_size=pool_size),
  Conv2D(64, filter_size, activation="relu"),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dropout(0.5),  
  Dense(10, activation='softmax'),
])


# Compile the model.
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=15,
  validation_data=(test_images, to_categorical(test_labels)),
  batch_size=128
)




#View First 30 Predictions
predictions = model.predict(test_images[:30])
print(np.argmax(predictions, axis=1))
print(test_labels[:30]) 

for i in range(30):
    pyplot.imshow(test_images[i], cmap='gray')
    pyplot.show()
    print("Model Prediction: [" + str(np.argmax(predictions, axis=1)[i]) + "]")
    print("True Label: [" + str(test_labels[i]) + "]") 
    





