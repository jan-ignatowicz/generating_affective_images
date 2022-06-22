import pandas as pd


def prepare_files(data, dataset):
    data_result = []
    for index, row in data.iterrows():
        data_result.append(
            {"dataset": dataset, "id": f"{row['Theme'].strip()}.jpg", "valence": row.Valence_mean,
             "arousal": row.Arousal_mean})

    return data_result


def prepare_oasis_file():
    data = pd.read_csv('../../datasets/unzipped/oasis/OASIS.csv')

    data["Arousal_mean"] = data["Arousal_mean"].map(lambda x: (x - 4) / 3)
    data["Valence_mean"] = data["Valence_mean"].map(lambda x: (x - 4) / 3)

    data_result = prepare_files(data, "OASIS")

    result = pd.DataFrame(data_result)

    result.to_csv("../csv_files/OASIS.csv", index=False)  # save file


if __name__ == '__main__':
    prepare_oasis_file()
