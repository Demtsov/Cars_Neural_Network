import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.models import Sequential
from tensorflow import keras
from keras.utils import get_file
dataset_dir=pathlib.Path("dataset/Priora_data/")

sound_count = len(list(dataset_dir.glob("*/*.mp3")))
print(f"Всего файлов: {sound_count}")
batch_size=32
train_ds = tf.keras.utils.audio_dataset_from_directory(
    dataset_dir,
	validation_split = 0.3,
	subset = "training",
	seed = 123,
	batch_size = batch_size
)

validation_ds = tf.keras.utils.audio_dataset_from_directory(
    dataset_dir,
	validation_split = 0.3,
	subset = "validation",
	seed = 123,
	batch_size = batch_size
)
class_names = train_ds.class_names
print(f"Class names: {class_names}")
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
model = Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))  # Замените размер и количество каналов на ваши данные
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))  # 2 класса для бинарной классификации, замените на количество классов в вашей задаче

model.compile(
	optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])
model.build()
model.summary()

epochs = 10
history = model.fit(
	train_ds,
	validation_data=validation_ds,
	epochs=epochs)

# visualize training and validation results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()






