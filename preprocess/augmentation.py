import os
import random

import pandas as pd
from PIL import Image, ImageChops, ImageFilter, ImageEnhance

import settings as opt


# Dataset augmentation

def horizontal_shift(img, xoffset):
    return ImageChops.offset(img, xoffset, 0)


def vertical_shift(img, yoffset):
    return ImageChops.offset(img, 0, yoffset)


def blur(img):
    return img.filter(ImageFilter.BLUR)


def contour(img):
    return img.filter(ImageFilter.CONTOUR)


def detail(img):
    return img.filter(ImageFilter.DETAIL)


def edge_enhance(img):
    return img.filter(ImageFilter.EDGE_ENHANCE)


def brighten(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)


def color(img, factor):
    return ImageEnhance.Color(img).enhance(factor)


def contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)


def rotate_90(img):
    return img.transpose(method=Image.ROTATE_90)


def crop(img, left, upper, right, lower):
    return img.crop((left, upper, right, lower)).resize((opt.image_size, opt.image_size))


def augment_image(image_name_full, augmented_images_path):
    img = Image.open(os.path.join(opt.ROOT_DIR, opt.all_images_path, image_name_full))
    img = img.convert('RGB')

    image_name, img_ext = os.path.splitext(image_name_full)

    save_img_path = augmented_images_path + image_name

    # save original images
    img.save(f'{save_img_path}_base.jpg', "JPEG")

    # augment and save (4 methods)
    img_detail = detail(img)
    img_detail.save(f'{save_img_path}_detail.jpg', "JPEG")

    img_edge_enhance = edge_enhance(img)
    img_edge_enhance.save(f'{save_img_path}_edgeenhance.jpg', "JPEG")

    img_bright = brighten(img, 1.3)
    img_bright.save(f'{save_img_path}_bright.jpg', "JPEG")

    img_bright2 = brighten(img, 0.9)
    img_bright2.save(f'{save_img_path}_bright2.jpg', "JPEG")

    img_rotate90 = rotate_90(img)
    img_rotate90.save(f'{save_img_path}_rotate90.jpg', "JPEG")

    img_rotate180 = rotate_90(img_rotate90)
    img_rotate180.save(f'{save_img_path}_rotate180.jpg', "JPEG")

    img_rotate270 = rotate_90(img_rotate180)
    img_rotate270.save(f'{save_img_path}_rotate270.jpg', "JPEG")


augmented_train_images_path = os.path.join(opt.ROOT_DIR,
                                           "datasets/all_data_affective/augmented_train/")
augmented_val_images_path = os.path.join(opt.ROOT_DIR, "datasets/all_data_affective/augmented_val/")
try:
    os.makedirs(augmented_train_images_path)
    os.makedirs(augmented_val_images_path)
except OSError:
    pass

data = pd.read_csv("csv_files/ALL_DATA.csv")
training_data = []
validation_data = []

images_list = os.listdir(os.path.join(opt.ROOT_DIR, opt.all_images_path))

assert len(images_list) == data.shape[0]

for i, image_name_path in enumerate(images_list):
    dataset, img_name = image_name_path.split("_", 1)

    row = data[(data["dataset"] == dataset) & (data["id"] == img_name)]
    category_id = row.category_id.values[0]

    image_name, img_ext = os.path.splitext(image_name_path)

    if i == 10:
        print(i)
        break

    if random.random() < 0.2:
        augment_image(image_name_path, augmented_val_images_path)

        validation_data.append(
            {"id": f"{image_name}_base{img_ext}", "label": category_id})
        validation_data.append(
            {"id": f"{image_name}_detail{img_ext}", "label": category_id})
        validation_data.append(
            {"id": f"{image_name}_edgeenhance{img_ext}", "label": category_id})
        validation_data.append(
            {"id": f"{image_name}_bright{img_ext}", "label": category_id})
        validation_data.append(
            {"id": f"{image_name}_bright2{img_ext}", "label": category_id})
        validation_data.append(
            {"id": f"{image_name}_rotate90{img_ext}", "label": category_id})
        validation_data.append(
            {"id": f"{image_name}_rotate180{img_ext}", "label": category_id})
        validation_data.append(
            {"id": f"{image_name}_rotate270{img_ext}", "label": category_id})

    else:
        augment_image(image_name_path, augmented_train_images_path)

        training_data.append(
            {"id": f"{image_name}_base{img_ext}", "label": category_id})
        training_data.append(
            {"id": f"{image_name}_detail{img_ext}", "label": category_id})
        training_data.append(
            {"id": f"{image_name}_edgeenhance{img_ext}", "label": category_id})
        training_data.append(
            {"id": f"{image_name}_bright{img_ext}", "label": category_id})
        training_data.append(
            {"id": f"{image_name}_bright2{img_ext}", "label": category_id})
        training_data.append(
            {"id": f"{image_name}_rotate90{img_ext}", "label": category_id})
        training_data.append(
            {"id": f"{image_name}_rotate180{img_ext}", "label": category_id})
        training_data.append(
            {"id": f"{image_name}_rotate270{img_ext}", "label": category_id})

dataset_path = os.path.join(opt.ROOT_DIR, "datasets/")

training_data = pd.DataFrame(training_data)
training_data.to_csv(f'{dataset_path}TrainAugmentedData.csv', index=False, header=False)

validation_data = pd.DataFrame(validation_data)
validation_data.to_csv(f'{dataset_path}ValidationAugmentedData.csv', index=False, header=False)
