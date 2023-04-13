import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers
from tensorflow.keras.datasets import mnist

# get the data set
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# num_classes = 10
num_classes =len(np.unique(train_labels))

model = tf.keras.Sequential([
  
  layers.Conv2D(8, 5, padding='valid', strides = (2,2),activation='relu', input_shape=(28, 28, 3)),

  layers.Conv2D(16, 3, padding='same', strides = (2,2),activation='relu'),

  layers.Conv2D(32, 3, padding='same', strides = (2,2),activation='relu'),

  layers.Conv2D(32, 3, padding='same', strides = (2,2),activation='relu'),
  
  #default AveragePooling2D(pool_size=(2, 2),strides=(1, 1), padding='valid'),
  layers.AveragePooling2D(),

  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes),
  layers.Softmax()
])
#print(model.summary())
#print(num_classes)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data = (test_images, test_labels))

#plot the accuracy
plt.title("CNN on Fasion MNIST data")
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label = 'testing accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim([0.2, 0.8])
plt.savefig('CNN.jpg')