import os
import zipfile

import utils


zipped_datasets_dir = os.path.join(utils.ROOT_DIR, 'datasets/zipped')
directory_to_extract_to = os.path.join(utils.ROOT_DIR, 'datasets/unzipped')

# # unpack zipped datasets
for filename in os.listdir(zipped_datasets_dir):
    f = os.path.join(zipped_datasets_dir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        # print(f)
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
