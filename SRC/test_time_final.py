import glob
from keras_segmentation.models.unet import vgg_unet
import time
import numpy as np

model480_name = './modele_256_384_480/480/vgg_unet_4classes_newLabels_model_21'
model480 = vgg_unet(n_classes=4, input_height=480, input_width=480)
model480.load_weights(model480_name)

## Analiza modelu
# model480.summary()
#print("Layers length: ")
#print(len(model480.layers))

##
model256_name = './modele_256_384_480/256/vgg_unet_4cl_newlabels_model_20.h5'
model256 = vgg_unet(n_classes=4, input_height=256, input_width=256)
model256.load_weights(model256_name)

##
model384_name = './modele_256_384_480/384/vgg_unet_4cl_newlabels_model_20.h5'
model384 = vgg_unet(n_classes=4, input_height=384, input_width=384)
model384.load_weights(model384_name)

##
models = [model256, model384, model480]
test_img_path = glob.glob('./PRZEKROJE_TEST/*.png')[:1]
test_iters = [1, 10, 50, 100, 500, 1000]
time_res = np.zeros((len(test_img_path), len(models), len(test_iters)))

out = models[0].predict_segmentation(inp=test_img_path[0])
for it_img, img_path in enumerate(test_img_path):
    for it_model, model in enumerate(models):
        for it_inference, test_iter in enumerate(test_iters):
            print("Now {} test iters".format(test_iter))
            start_time = time.time()
            for i in range(0, test_iter):
                out = model.predict_segmentation(inp=img_path)
            time_res[it_img, it_model, it_inference] = (time.time() - start_time) / test_iter

print(time_res)
file_path = "./results_final.npy"
np.save(file_path, [time_res, test_iters])
