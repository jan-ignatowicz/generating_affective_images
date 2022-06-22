import os

import numpy as np
import pandas as pd

import settings as opt

csv_files_path = os.path.join(opt.ROOT_DIR, "preprocess/csv_files/")
dataset_path = os.path.join(opt.ROOT_DIR, "datasets/")

data_emomadrid = pd.read_csv(f'{csv_files_path}EMOMADRID.csv')
data_gaped = pd.read_csv(f'{csv_files_path}GAPED.csv')
data_iaps = pd.read_csv(f'{csv_files_path}IAPS.csv')
data_naps = pd.read_csv(f'{csv_files_path}NAPS.csv')
data_oasis = pd.read_csv(f'{csv_files_path}OASIS.csv')
data_sfip = pd.read_csv(f'{csv_files_path}SFIP.csv')

all_data = pd.concat(
    [data_emomadrid, data_gaped, data_iaps, data_naps, data_oasis, data_sfip],
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    copy=True,
)


def calc_angle(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


data_result = []
for index, row in all_data.iterrows():
    angle = calc_angle((0, 0), (row.valence, row.arousal))

    if abs(row.valence) <= 0.25 and abs(row.arousal) <= 0.25:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "neutral", "category_id": 0})
    elif 0 <= angle < 30:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "content", "category_id": 1})
    elif 30 <= angle < 60:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "relaxed", "category_id": 2})
    elif 60 <= angle < 90:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "calm", "category_id": 3})
    elif 90 <= angle < 120:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "tired", "category_id": 4})
    elif 120 <= angle < 150:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "bored", "category_id": 5})
    elif 150 <= angle < 180:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "depressed", "category_id": 6})
    elif 180 <= angle < 210:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "frustrated", "category_id": 7})
    elif 210 <= angle < 240:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "angry", "category_id": 8})
    elif 240 <= angle < 270:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "tense", "category_id": 9})
    elif 270 <= angle < 300:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "excited", "category_id": 10})
    elif 300 <= angle < 330:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "delighted", "category_id": 11})
    elif 330 <= angle < 360:
        data_result.append({"dataset": row.dataset, "id": row.id, "valence": row.valence,
                            "arousal": row.arousal, "category": "happy", "category_id": 12})
    else:
        print("Angle not in range!")

result = pd.DataFrame(data_result)
result.to_csv(f'{csv_files_path}ALL_DATA.csv', index=False)  # save file

training_data = []
for index, row in result.iterrows():
    training_data.append({"id": f"{row.dataset}_{row.id}", "label": row.category_id})

training_data_result = pd.DataFrame(training_data)
training_data_result.to_csv(f'{dataset_path}TrainingData.csv', index=False,
                            header=False)  # save file

# Augmented data
augmented_data = []
for index, row in result.iterrows():
    image_name, img_ext = os.path.splitext(row.id)
    augmented_data.append(
        {"id": f"{row.dataset}_{image_name}_base{img_ext}", "label": row.category_id})
    augmented_data.append(
        {"id": f"{row.dataset}_{image_name}_detail{img_ext}", "label": row.category_id})
    augmented_data.append(
        {"id": f"{row.dataset}_{image_name}_edgeenhance{img_ext}", "label": row.category_id})
    augmented_data.append(
        {"id": f"{row.dataset}_{image_name}_bright{img_ext}", "label": row.category_id})
    augmented_data.append(
        {"id": f"{row.dataset}_{image_name}_bright2{img_ext}", "label": row.category_id})
    augmented_data.append(
        {"id": f"{row.dataset}_{image_name}_rotate90{img_ext}", "label": row.category_id})
    augmented_data.append(
        {"id": f"{row.dataset}_{image_name}_rotate180{img_ext}", "label": row.category_id})
    augmented_data.append(
        {"id": f"{row.dataset}_{image_name}_rotate270{img_ext}", "label": row.category_id})

augmented_data_result = pd.DataFrame(augmented_data)
augmented_data_result.to_csv(f'{dataset_path}AugmentedTrainingData.csv', index=False,
                             header=False)  # save file
