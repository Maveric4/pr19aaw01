from __future__ import absolute_import, division, print_function, unicode_literals
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix
import glob
import cv2
import numpy as np

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt
import json

"""## Download the Oxford-IIIT Pets dataset
The dataset is already included in TensorFlow datasets, all that is needed to do is download it. The segmentation masks are included in version 3.0.0, which is why this particular version is used.
"""
# Daniel: Bez uczenia modelu, tylko ewaluacja
load_trained_model = True
IMG_SHAPE = 128

# Daniel: ja już pobrałem te pliki, są na repo, bo jak próbowałem odpalić z download=True to wywalało mi jakieś błędy nie do naprawienia
dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True, download=False)

"""The following code performs a simple augmentation of flipping an image. In addition,  image is normalized to [0,1]. 
Finally, as mentioned above the pixels in the segmentation mask are labeled either {1, 2, 3}. 
For the sake of convinience, let's subtract 1 from the segmentation mask, resulting in labels that are : {0, 1, 2}."""

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (IMG_SHAPE, IMG_SHAPE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SHAPE, IMG_SHAPE))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (IMG_SHAPE, IMG_SHAPE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SHAPE, IMG_SHAPE))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

"""The dataset already contains the required splits of test and train and so let's continue to use the same split."""

TRAIN_LENGTH = info.splits['train'].num_examples
# BATCH_SIZE = 8
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

"""Let's take a look at an image example and it's correponding mask from the dataset."""

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Groundtruth Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.gray()
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
# display([sample_image, sample_mask])

OUTPUT_CHANNELS = 3

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def mask_floodFilling(mask):
    ret, thresh = cv2.threshold(np.array(mask), 1.5, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((3, 3), np.uint8)
    # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    img_blur = cv2.medianBlur(closing, 3)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cont = cv2.drawContours(thresh, contours, -1, (0, 255, 0), 3)

    # Copy the thresholded image.
    im_floodfill = img_blur.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    retval, im_floodfill, mask, rect = cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill.astype('uint8'))

    # # Combine the two images to get the foreground.
    im_out = thresh.astype('uint8') | im_floodfill_inv

    return np.array(im_out).reshape((128, 128, 1))

def mask_cnts(mask):
    ret, thresh = cv2.threshold(np.array(mask), 1.5, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cont = cv2.drawContours(thresh, contours, -1, (0, 255, 0), 3)

    return np.array(thresh).reshape((128, 128, 1))

def show_predictions(dataset=None, num=1):
    if dataset is not None:
        for image, mask in dataset.take(num):
            image_sep = image[0]
            mask_sep = mask[0]
            groundtruth = mask_floodFilling(mask_sep)
            # groundtruth = mask_cnts(mask_sep)
            display([image[0], mask[0], groundtruth])
    else:
        print("Error!")



show_predictions(test_dataset, 5)
