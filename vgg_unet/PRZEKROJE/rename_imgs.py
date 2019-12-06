import glob
import os

for it, img_path in enumerate(glob.glob('./*.png')):
    splitted = img_path.split('.PNG')
    new_name = ''.join(splitted + ['.png'])
    # print(new_name)
    # print(''.join(splitted + ['.png']))

    # print(img_path)
    # print(old_name)
    os.rename(img_path, new_name)



