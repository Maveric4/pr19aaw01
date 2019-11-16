from __future__ import absolute_import, division, print_function, unicode_literals
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix

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

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

"""## Define the model
The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). 
In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. 
Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, 
and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial]
(https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py). 

The reason to output three channels is because there are three possible labels for each pixel. 
Think of this as multi-classification where each pixel is being classified into three classes.
"""
OUTPUT_CHANNELS = 3
"""As mentioned, the encoder will be a pretrained MobileNetV2 model which is prepared and ready to use in 
[tf.keras.applications](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications). 
The encoder consists of specific outputs from intermediate layers in the model. 
Note that the encoder will not be trained during the training process."""

base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_SHAPE, IMG_SHAPE, 3], include_top=False)
# base_model.summary()

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

"""The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples."""

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    # pix2pix.upsample(32, 3),   # 64x64 -> 128x128
    # pix2pix.upsample(16, 3),   # 128x128 -> 256x256
]

def unet_model(output_channels):

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same', activation='softmax')  #256x256 -> 512x512

  inputs = tf.keras.layers.Input(shape=[IMG_SHAPE, IMG_SHAPE, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

"""## Train the model
Now, all that is left to do is to compile and train the model. The loss being used here is losses.sparse_categorical_crossentropy. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and losses.sparse_categorical_crossentropy is the recommended loss for such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.
"""

model = unet_model(OUTPUT_CHANNELS)
base_model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""Have a quick look at the resulting model architecture:"""

tf.keras.utils.plot_model(model, show_shapes=True)

"""Let's try out the model to see what it predicts before training."""

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      print(image.shape)
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

"""Let's observe how the model improves while it is training. To accomplish this task, a callback function is defined below."""

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

# Loading trained model and previous loss and accuracy values

if load_trained_model:
    model = tf.keras.models.load_model('my_model.h5')
    history_dict = json.load(open("./history_dict.json", 'r'))
    history_dict = eval(history_dict)
else:
    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset,
                              callbacks=[DisplayCallback()])

    history_dict = model_history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig("loss.jpg")
plt.show()


plt.figure()
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.savefig("acc.jpg")
plt.show()

## Save model
if not load_trained_model:
    model.save("new_model.h5")

    # Saving history
    # Get the dictionary containing each metric and the loss for each epoch
    # history_dict = model_history.history
    # Save it under the form of a json file
    json.dump(history_dict, open("./new_history_dict.json", 'w'))

"""## Make predictions

Let's make some predictions. In the interest of saving time, the number of epochs was kept small, but you may set this higher to achieve more accurate results.
"""

show_predictions(test_dataset, 3)
"""## Next steps
Now that you have an understanding of what image segmentation is and how it works, you can try this tutorial out with different intermediate layer outputs, or even different pretrained model. You may also challenge yourself by trying out the [Carvana](https://www.kaggle.com/c/carvana-image-masking-challenge/overview) image masking challenge hosted on Kaggle.

You may also want to see the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) for another model you can retrain on your own data.
"""

import glob
import cv2
import numpy as np
it = 0
for name in glob.glob("../images/*.jpg"):
    img = cv2.imread(name)
    img = tf.image.resize(img, (IMG_SHAPE, IMG_SHAPE))
    img = tf.cast(img, tf.float32) / 255.0
    img = np.array(img).reshape((IMG_SHAPE, IMG_SHAPE, 3))
    pred_mask = model.predict(img[tf.newaxis, ...])
    display([img, create_mask(pred_mask)])
    it += 1
    if it == 5:
        break

for name in glob.glob("../PRZEKROJE/*.PNG"):
    img = cv2.imread(name)
    img = tf.image.resize(img, (IMG_SHAPE, IMG_SHAPE))
    img = tf.cast(img, tf.float32) / 255.0
    img = np.array(img).reshape((IMG_SHAPE, IMG_SHAPE, 3))
    pred_mask = model.predict(img[tf.newaxis, ...])
    display([img, create_mask(pred_mask)])