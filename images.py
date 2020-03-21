# Zindi UmojaHack Hackathon
# Image Classification for Deep Sea Invertebrates
# TF Implementation


from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pathlib

print('Using TF version == ' ,tf.__version__)

data_dir = pathlib.Path('inv/UmojaHack_1_invertebrates/train_small')


image_count = len(list(data_dir.glob('*/*.jpeg')))

print(f'The image count is {image_count}')


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "submission.csv"])


list_ds = tf.data.Dataset.list_files(str(data_dir))

for f in list_ds.take(5):
  print(f.numpy())

lim = list(data_dir.glob('Limopsis_chuni/*'))

for image_path in lim[:3]:
    display.display(Image.open(str(image_path)))

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

