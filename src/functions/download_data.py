import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from s3fs import S3FileSystem


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def get_golden_paths(
    from_s3: bool,
    task: str,
    source: str,
    dep: str,
    year: str,
    tiles_size: str,
) -> Tuple[List[str], List[str]]:
    """
    Get paths to images and labels making up a golden dataset of 32 observations
    from s3 or local.

    Args:
        from_s3 (bool): True if data should be downloaded from s3, False otherwise.
        task (str): Task.
        source (str): Satellite source.
        dep (str): Department.
        year (str): Year.
        tiles_size (str): Tiles size.

    Returns:
        Tuple[List[str], List[str]]: Paths to patchs and labels.
    """
    if from_s3:
        fs = get_file_system()

        patchs = fs.ls(
            (
                f"projet-slums-detection/golden-test/patchs/"
                f"{task}/{source}/{dep}/{year}/{tiles_size}"
            )
        )
        labels = fs.ls(
            (
                f"projet-slums-detection/golden-test/labels/"
                f"{task}/{source}/{dep}/{year}/{tiles_size}"
            )
        )
    else:
        patchs_path = (
            f"data/data-preprocessed/golden-test/patchs/"
            f"{task}/{source}/{dep}/{year}/{tiles_size}"
        )
        labels_path = (
            f"data/data-preprocessed/golden-test/labels/{task}/{source}/{dep}/{year}/{tiles_size}"
        )

        patch_cmd = [
            "mc",
            "cp",
            "--quiet",
            "-r",
            f"s3/projet-slums-detection/golden-test/patchs/"
            f"{task}/{source}/{dep}/{year}/{tiles_size}/",
            patchs_path + "/",
        ]
        label_cmd = [
            "mc",
            "cp",
            "--quiet",
            "-r",
            f"s3/projet-slums-detection/golden-test/labels/"
            f"{task}/{source}/{dep}/{year}/{tiles_size}/",
            labels_path + "/",
        ]
        with open("/dev/null", "w") as devnull:
            # download patchs
            subprocess.run(patch_cmd, check=True, stdout=devnull, stderr=devnull)
            # download labels
            subprocess.run(label_cmd, check=True, stdout=devnull, stderr=devnull)

        patchs = [
            f"{patchs_path}/{filename}"
            for filename in os.listdir(patchs_path)
            if Path(filename).suffix != ".yaml"
        ]
        labels = [f"{labels_path}/{filename}" for filename in os.listdir(labels_path)]

    return patchs, labels


def get_patchs_labels(
    from_s3: bool,
    task: str,
    source: str,
    dep: str,
    year: str,
    tiles_size: str,
    type_labeler: str,
    train: bool,
) -> Tuple[List[str], List[str]]:
    """
    Get paths to patches and labels from s3 or local.

    Args:
        from_s3 (bool): True if data should be downloaded from s3, False otherwise.
        task (str): Task.
        source (str): Satellite source.
        dep (str): Department.
        year (str): Year.
        tiles_size (str): Tiles size.
        type_labeler (str): Type of labeler.
        train (bool): True if data should be downloaded for training, False otherwise.

    Returns:
        Tuple[List[str], List[str]]: Paths to patchs and labels.
    """
    if train:
        stage = "train"
    else:
        stage = "test"

    if from_s3:
        fs = get_file_system()

        patchs = fs.ls(
            (
                f"projet-slums-detection/data-preprocessed/patchs/"
                f"{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/{stage}"
            )
        )

        labels = fs.ls(
            (
                f"projet-slums-detection/data-preprocessed/labels/"
                f"{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/{stage}"
            )
        )

    else:
        patchs_path = (
            f"data/data-preprocessed/patchs/" f"{task}/{source}/{dep}/{year}/{tiles_size}/{stage}"
        )
        labels_path = (
            f"data/data-preprocessed/labels/"
            f"{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/{stage}"
        )

        download_data(
            patchs_path,
            labels_path,
            task,
            source,
            dep,
            year,
            tiles_size,
            type_labeler,
            train,
        )

        patchs = [
            f"{patchs_path}/{filename}"
            for filename in os.listdir(patchs_path)
            if Path(filename).suffix != ".yaml"
        ]
        labels = [f"{labels_path}/{filename}" for filename in os.listdir(labels_path)]

    return patchs, labels


def download_data(
    patchs_path: str,
    labels_path: str,
    task: str,
    source: str,
    dep: str,
    year: str,
    tiles_size: str,
    type_labeler: str,
    train: bool,
) -> None:
    """
    Download data for a specific context, if not already downloaded.

    Args:
        patchs_path (str): Paths to patchs.
        labels_path (str): Paths to labels.
        task (str): Task.
        source (str): Satellite source.
        dep (str): Department.
        year (str): Year.
        tiles_size (str): Tiles size.
        type_labeler (str): Type of labeler.
        train (bool): True if data should be downloaded for training, False otherwise.
    """
    if train:
        stage = "train"
    else:
        stage = "test"

    all_exist = all(os.path.exists(f"{directory}") for directory in [patchs_path, labels_path])

    if all_exist:
        return None

    patch_cmd = [
        "mc",
        "cp",
        "-r",
        f"s3/projet-slums-detection/data-preprocessed/patchs/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/{stage}/",  # noqa
        f"data/data-preprocessed/patchs/{task}/{source}/{dep}/{year}/{tiles_size}/{stage}/",
    ]

    label_cmd = [
        "mc",
        "cp",
        "-r",
        f"s3/projet-slums-detection/data-preprocessed/labels/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/{stage}/",  # noqa
        f"data/data-preprocessed/labels/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/{stage}/",
    ]

    print("Downloading data from S3...\n")
    with open("/dev/null", "w") as devnull:
        # download patchs
        subprocess.run(patch_cmd, check=True, stdout=devnull, stderr=devnull)
        # download labels
        subprocess.run(label_cmd, check=True, stdout=devnull, stderr=devnull)
    print("Downloading finished!\n")


def normalization_params(
    task: str, source: str, dep: str, year: str, tiles_size: str, type_labeler: str
):
    """
    Get normalization params from s3.

    task (str): Task.
    source (str): Satellite source.
    dep (str): Department.
    year (str): Year.
    tiles_size (str): Tiles size.
    type_labeler (str): Type of labeler.
    """
    params_path = f"data/data-preprocessed/patchs/{task}/{source}/{dep}/{year}/{tiles_size}/train/metrics-normalization.yaml"  # noqa
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params["mean"], params["std"]


def pooled_std_dev(n: List[int], means: List[float], std_devs: List[float]):
    """
    Computes the pooled standard deviation of all the data given
    the means, standard deviations, and number of observations
    of n subsets of data.

    Args:
        n (List[int]): List of the number of observations in each subset of data.
        means (List[float]): List of the means of each subset of data.
        std_devs (List[float]): List of the standard deviations of each subset of data.

    Returns:
        The pooled standard deviation of all the data.
    """
    n_total = np.sum(n)
    x_total = np.sum([ni * xi for ni, xi in zip(n, means)]) / n_total
    var_total = np.sum([(ni - 1) * si**2 for ni, si in zip(n, std_devs)]) + np.sum(
        [ni * (xi - x_total) ** 2 for ni, xi in zip(n, means)]
    )
    pooled_std_dev = np.sqrt(var_total / (n_total - 1))
    return pooled_std_dev
