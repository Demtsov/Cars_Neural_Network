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

sound_count = len(list(dataset_dir.glob("*/*.wav")))
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

input_shape1=(32,44100,2)
model = keras.models.Sequential()

# Добавьте слои для обработки аудиоданных
# Например, можно использовать сверточные и рекуррентные слои
model.add(keras.layers.Conv2D(16, (1, 100), activation='relu', input_shape=input_shape1))
model.add(keras.layers.MaxPooling2D((4, 4)))
model.add(keras.layers.Conv2D(64, (1, 100), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64,input_shape=(input_shape1,) ,activation='relu'))

# Добавьте выходной слой для классификации (зависит от задачи)
# Например, для задачи классификации звуковых классов
num_classes = 10
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(
	optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])
model.summary()