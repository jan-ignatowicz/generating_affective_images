import os
import shutil

from PIL import Image

all_images_path = "../datasets/all_images/"

# preprare EMOMADRID images
images_path = "../datasets/merged/EMOMADRID_ALL_IMAGES/"
images = os.listdir(images_path)
[shutil.copy(images_path + i, all_images_path + 'EMOMADRID_' + i) for i in images]

# preprare GAPED images
images_path = "../datasets/merged/GAPED_ALL_IMAGES/"
images = os.listdir(images_path)
[Image.open(images_path + i).convert("RGB").save(
    all_images_path + 'GAPED_' + f'{i.split(sep=".")[0]}.jpg', "JPEG") for i in images]

# preprare IAPS images
images_path = "../datasets/merged/IAPS_ALL_IMAGES/"
images = os.listdir(images_path)
[shutil.copy(images_path + i, all_images_path + 'IAPS_' + i) for i in images]

# preprare NAPS images
images_path = "../datasets/merged/NAPS_ALL_IMAGES/"
images = os.listdir(images_path)
[shutil.copy(images_path + i, all_images_path + 'NAPS_' + i) for i in images]

# preprare OASIS images
images_path = "../datasets/merged/OASIS_ALL_IMAGES/"
images = os.listdir(images_path)
[shutil.copy(images_path + i, all_images_path + 'OASIS_' + i) for i in images]

# preprare SFIP images
images_path = "../datasets/merged/SFIP_ALL_IMAGES/"
images = os.listdir(images_path)
[shutil.copy(images_path + i, all_images_path + 'SFIP_' + i) for i in images]
