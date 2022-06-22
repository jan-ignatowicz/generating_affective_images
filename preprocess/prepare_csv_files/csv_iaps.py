import pandas as pd

data = pd.read_excel(
    '../datasets/unzipped/OneDrive_1_16-03-2021/IAPS. International Affective Picture System/IAPS. Tech Report/AllSubjects_1-20.ods',
    engine='odf')

data.drop_duplicates(subset=['IAPS'], inplace=True)

data["aromn"] = data["aromn"].map(lambda x: (x - 5) / 4)
data["valmn"] = data["valmn"].map(lambda x: (x - 5) / 4)

data_result = []

for index, row in data.iterrows():
    id_name = str(row.IAPS)
    if str(row.IAPS).endswith("0"):
        id_name = str(row.IAPS)[:-2]
    data_result.append(
        {"dataset": "IAPS", "id": f"{id_name}.jpg", "valence": row.valmn,
         "arousal": row.aromn})

result = pd.DataFrame(data_result)
result.to_csv("../csv_files/IAPS.csv", index=False)  # save file
