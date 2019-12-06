import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
nbr_classes = 5

# for it, img_path in enumerate(glob.glob('./*.png')):
#     img = cv2.imread(img_path)
#     print(np.unique(img))
#     print(img.shape)
#     # print(cv2.countNonZero(np.array(img == 0, dtype=np.int)))
#     print(np.sum(np.array(img == 0, dtype=np.int)))
#     print(np.sum(np.array(img == 1, dtype=np.int)))
#     print(np.sum(np.array(img == 2, dtype=np.int)))
#     print(np.sum(np.array(img == 3, dtype=np.int)))
#     print(np.sum(np.array(img == 4, dtype=np.int)))
#     plt.figure()
#     plt.imshow(img * int(255.0/nbr_classes))
#     plt.show()
#     # cv2.imshow('image', img * 255.0/nbr_classes)
#     # cv2.waitKey(0)
#     if it > 3:
#         break

# for it, img_path in enumerate(glob.glob('./*.png')):
#     img = cv2.imread(img_path)
#     # for x in range(0, img.shape[0]):
#     #     for y in range(0, img.shape[1]):
#     #         if img[x, y, 0] == 0:
#     #             img[x, y, :] = 3
#
#     mask = np.array(img == 0, dtype=np.uint8)
#     mask *= 2
#     img += mask
#     # plt.figure()
#     # plt.imshow(img * int(255.0/nbr_classes))
#     # plt.show()
#     cv2.imwrite(img_path, img)
#     # if it > 3:
#     #     break

for it, img_path in enumerate(glob.glob('./*.png')):
    img = cv2.imread(img_path)
    # print(np.unique(img))
    img -= 1
    # print(np.unique(img))
    cv2.imwrite(img_path, img)