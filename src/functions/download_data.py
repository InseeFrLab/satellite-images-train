import os
import subprocess
from typing import List, Tuple
from pathlib import Path

from s3fs import S3FileSystem
import yaml


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


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

    # download patchs
    subprocess.run(patch_cmd, check=True)
    # download labels
    subprocess.run(label_cmd, check=True)


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
