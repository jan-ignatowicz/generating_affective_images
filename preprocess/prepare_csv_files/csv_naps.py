import pandas as pd

data = pd.read_excel(
    '../datasets/unzipped/OneDrive_1_16-03-2021/NAPS. Nencki Affective Picture System/Valence, arousal and approach-avoidance ratings (Marchewka et al., 2014).xls',
    engine='xlrd', skiprows=2)

data["M.8"] = data["M.8"].map(lambda x: (x - 5) / 4)
data["M.6"] = data["M.6"].map(lambda x: (x - 5) / 4)

data_result = []
for index, row in data.iterrows():
    data_result.append(
        {"dataset": "NAPS", "id": f"{row.ID}.jpg", "valence": row["M.6"], "arousal": row["M.8"]})

result = pd.DataFrame(data_result)
result.to_csv("../csv_files/NAPS.csv", index=False)  # save file
