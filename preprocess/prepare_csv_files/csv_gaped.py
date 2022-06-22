"""This file preprocess GAPED dataset, it merges the txt files to one csv file"""

import os
import re

import pandas as pd

import settings

pattern = "^[a-zA-Z]{1,2}.txt$"

image_file_ext = "bmp"

gaped_dir = os.path.join(settings.ROOT_DIR, "datasets/unzipped/GAPED")

# prepare GAPED csv file
data_result = []
for filename in os.listdir(gaped_dir):

    if re.search(pattern, filename):
        with open(os.path.join(gaped_dir, filename)) as f:
            lines = f.readlines()

            for row in lines[1:]:
                valence = float(row.split()[1]) / 50 - 1
                arousal = float(row.split()[2]) / 50 - 1

                data_result.append(
                    {"dataset": "GAPED", "id": f"{row.split()[0]}", "valence": valence,
                     "arousal": arousal})

result = pd.DataFrame(data_result)
result.to_csv("../csv_files/GAPED.csv", index=False)  # save file
