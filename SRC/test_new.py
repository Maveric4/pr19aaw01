import glob
from keras_segmentation.models.unet import vgg_unet
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

model = vgg_unet(n_classes=4, input_height=480, input_width=480)

model_name = './modele_256_384_480/480/vgg_unet_4classes_newLabels_model_21'
model.load_weights(model_name)
model.summary()
print("Number of layers:")
print(len(model.layers))

for img_path in glob.glob('./PRZEKROJE_TEST/*.png'):
    out = model.predict_segmentation(inp=img_path)
    if not os.path.isdir('./por/'):
        os.mkdir('./por/')
    cv2.imwrite('./por/' + os.path.split(img_path)[-1], out)
    print(np.unique(out))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
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

    plt.gray()
    ax1.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original image')
    ax2.imshow(annot_img)
    ax2.set_title('Groundtruth image')
    ax3.imshow(predicted)
    ax3.set_title('Predicted image')
    plt.savefig('./por/' + os.path.split(img_path)[-1])
    plt.show()


