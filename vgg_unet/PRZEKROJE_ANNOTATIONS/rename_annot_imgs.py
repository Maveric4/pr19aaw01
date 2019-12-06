import glob
import os

# for it, img_path in enumerate(glob.glob('./*.png')):

# for it, nbr in enumerate(range(0, len(glob.glob('./*.png')))):
#     new_name = 'K_000_0{:02d}.PNG'.format(it)
#     old_name = './Label_{}.png'.format(it+2)
#     print(new_name)
#     print(old_name)
#     os.rename(old_name, new_name)

for it, img_path in enumerate(glob.glob('./*.png')):
    splitted = img_path.split('.PNG')
    new_name = ''.join(splitted + ['.png'])
    # print(new_name)
    # print(''.join(splitted + ['.png']))

    # print(img_path)
    # print(old_name)
    os.rename(img_path, new_name)



