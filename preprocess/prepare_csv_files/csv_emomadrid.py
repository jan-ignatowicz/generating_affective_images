import pandas as pd

data = pd.read_excel('../datasets/unzipped/emomadrid/EMindex.xlsx', engine='openpyxl', skiprows=1)

for i, o in enumerate(data["Thumbnail"]):
    if o == "Empty":
        data.drop(i, inplace=True)

data = data[["EM CODE", "Mean Arousal", "Mean Valence"]]
data["Mean Arousal"].astype(float)

data["Mean Arousal"] = data["Mean Arousal"].div(2)
data["Mean Valence"] = data["Mean Valence"].div(2)

data_result = []
for index, row in data.iterrows():
    data_result.append(
        {"dataset": "EMOMADRID", "id": f"{row['EM CODE']}.jpg", "valence": row["Mean Valence"],
         "arousal": row["Mean Arousal"]})

result = pd.DataFrame(data_result)
result.to_csv("../csv_files/EMOMADRID.csv", index=False)  # save file
