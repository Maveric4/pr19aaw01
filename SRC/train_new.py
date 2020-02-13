import os
from keras_segmentation.models.unet import vgg_unet
model = vgg_unet(n_classes=4, input_height=480, input_width=480)

if os.path.isdir('./chckpt/'):
    os.mkdir('./chckpt/')

model.train(
    train_images="./PRZEKROJE/",
    train_annotations="./PRZEKROJE_ANNOTATIONS/",
    checkpoints_path="./chckpt/vgg_unet_4classes_newLabels",
    epochs=25,
    batch_size=1,
    val_images="./PRZEKROJE_VAL/",
    val_annotations="./PRZEKROJE_ANNOTATIONS_VAL/",
    val_batch_size=1,
    validate=True,
    do_augment=True
)