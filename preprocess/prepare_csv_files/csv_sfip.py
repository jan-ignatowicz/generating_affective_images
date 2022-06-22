import os

import pandas as pd

data = pd.read_excel(
    '../datasets/unzipped/OneDrive_1_16-03-2021/NAPS. Nencki Affective Picture System/SFIP ratings (Michałowski et al., 2016).xls',
    engine='xlrd', skiprows=3)

data["arousal"] = ((data["CT_M_A"] + data["BL_M_A"] + data["SP_M_A"] + data["SO_M_A"]) / 4).map(
    lambda x: (x - 5) / 4)
data["valence"] = ((data["CT_M_V"] + data["BL_M_V"] + data["SP_M_V"] + data["SO_M_V"]) / 4).map(
    lambda x: (x - 5) / 4)

# delete duplicates rows
sfip_images = os.listdir("../../datasets/merged/SFIP_ALL_IMAGES/")
sfip_images_id = [i.split(sep='.')[0] for i in sfip_images]
for id in data["SFIP NAME"]:
    if id not in sfip_images_id:
        data.drop(data.loc[data['SFIP NAME'] == id].index, inplace=True)

data_result = []

for index, row in data.iterrows():
    data_result.append(
        {"dataset": "SFIP", "id": f"{row['SFIP NAME']}.jpg", "valence": row["valence"],
         "arousal": row["arousal"]})

result = pd.DataFrame(data_result)
result.to_csv("../csv_files/SFIP.csv", index=False)  # save file
