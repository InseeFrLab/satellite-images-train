# %env AWS_ACCESS_KEY_ID=projet-slums-key
# %env AWS_SECRET_ACCESS_KEY=4mmZ9gLZUl2Jm5Tf9ALkCjqSaxfHhK7U

import os

import numpy as np
from astrovision.data import SatelliteImage
from tqdm import tqdm

from functions import download_data

del os.environ["AWS_SESSION_TOKEN"]


source = "PLEIADES"
dep = "GUADELOUPE"
year = "2020"
n_bands = 3
type_labeler = "BDTOPO"
task = "segmentation"
tiles_size = 250

fs = download_data.get_file_system()

mean_dic = {}
for path_dep in fs.ls(f"projet-slums-detection/data-preprocessed/patchs/{task}/{source}/"):
    print(path_dep)
    dep = path_dep.split("/")[-1]
    mean_dic[dep] = {}
    for path_year in fs.ls(path_dep):
        print(year)
        year = path_year.split("/")[-1]
        temp_arr = np.empty((0, 3))
        for im in tqdm(fs.ls(f"{path_year}/{tiles_size}")):
            si = SatelliteImage.from_raster(
                file_path=f"s3://{im}",
                dep=None,
                date=None,
                n_bands=3,
                cast_to_float=False,
            )
            temp_arr = np.vstack((temp_arr, np.mean(si.array, axis=(1, 2))))
        mean_dic[dep][year] = np.mean(temp_arr, axis=0)
