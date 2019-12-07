
from keras_segmentation.models.unet import vgg_unet
model = vgg_unet(n_classes=4, input_height=480, input_width=480)


model.train(
    train_images="./PRZEKROJE/",
    train_annotations="./PRZEKROJE_ANNOTATIONS/",
    checkpoints_path="./vgg_unet_4cl_newlabels",
    epochs=25,
    batch_size=1,
    val_images="./PRZEKROJE_VAL/",
    val_annotations="./PRZEKROJE_ANNOTATIONS_VAL/",
    val_batch_size=1,
    validate=True,
    do_augment=True
)