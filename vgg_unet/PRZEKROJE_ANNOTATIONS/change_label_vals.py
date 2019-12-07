import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


def change_val(img):

    # labels_inside = img == [0, 255, 0]
    labels_inside = img[:, :, 1] == 255

    labels_background = (img[:, :, 0] == 0) * (img[:, :, 1] == 0) * (img[:, :, 2] == 0)
    labels_outside = img[:, :, 0] == 255

    labels_bad = img[:, :, 2] == 255

    plt.figure()
    plt.gray()
    labels_inside = labels_inside * 2
    labels_background = labels_background * 0
    labels_outside = labels_outside * 1
    labels_bad = labels_bad * 3
    gt = np.zeros((img.shape[0], img.shape[1]))
    gt = labels_inside + labels_background + labels_bad + labels_outside
    plt.imshow(gt * 255.0 / 3)
    plt.show()

    return gt


for it, img_path in enumerate(glob.glob("./*.png")):
    img = cv2.imread(img_path)
    print(np.unique(img))
    print(img_path)
    print(img.shape)

    plt.figure()
    plt.gray()
    plt.imshow(img[:, :, 2]*255.0/3)
    plt.show()

    # # plt.figure()
    # color = ('b', 'g', 'r')
    # for i, col in enumerate(color):
    #     plt.figure()
    #     histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    #     print(histr)
    #     print()
    #     print()
    #     plt.plot(histr, color=col)
    #     plt.xlim([-5, 260])
    #     plt.show()

    # gt = change_val(img)

    # cv2.imwrite(img_path, gt)

    if it > 1:
        break