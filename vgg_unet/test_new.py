import glob
from keras_segmentation.models.unet import vgg_unet
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = vgg_unet(n_classes=4, input_height=480, input_width=480)
model_name = 'vgg_unet_4classes_model_17'
# model_name = 'psp_4classes_model_24'
model.load_weights(model_name)
model.summary()

for img_path in glob.glob('./PRZEKROJE_TEST/*.png'):
    out = model.predict_segmentation(inp=img_path)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Horizontally stacked subplots')
    print(img_path)
    orig = cv2.imread(img_path)
    annot_path = img_path.replace("PRZEKROJE_TEST", "PRZEKROJE_ANNOTATIONS_TEST")
    print(annot_path)

    annot_img = cv2.imread(annot_path)
    annot_unique_values = np.unique(annot_img).shape[0]
    annot_img = annot_img*np.uint8(255.0 / (annot_unique_values-1))
    predicted_unique_values = np.unique(out).shape[0]
    predicted_img = out*np.uint8(255.0 / (predicted_unique_values-1))
    predicted = cv2.resize(predicted_img, (orig.shape[0], orig.shape[1]), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)

    # predicted = cv2.resize(np.array(out*int(255.0/4), dtype=np.uint8), (608, 416))
    plt.gray()
    ax1.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original image')
    ax2.imshow(annot_img)
    ax2.set_title('Groundtruth image')
    ax3.imshow(predicted)
    ax3.set_title('Predicted image')
    plt.show()

