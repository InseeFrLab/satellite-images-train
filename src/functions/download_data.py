import os

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


def get_patchs_labels(
    from_s3: bool,
    task: str,
    source: str,
    dep: str,
    year: str,
    tiles_size: str,
    type_labeler: str,
):
    if from_s3:
        fs = get_file_system()

        patchs = fs.ls(
            (
                f"projet-slums-detection/data-preprocessed/patchs/"
                f"{task}/{source}/{dep}/{year}/{tiles_size}"
            )
        )

        labels = fs.ls(
            (
                f"projet-slums-detection/data-preprocessed/labels/"
                f"{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}"
            )
        )

    else:
        patchs_path = f"data/data-preprocessed/patchs/" f"{task}/{source}/{dep}/{year}/{tiles_size}"
        labels_path = (
            f"data/data-preprocessed/labels/"
            f"{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}"
        )

        download_data(patchs_path, labels_path, task, source, dep, year, tiles_size, type_labeler)

        patchs = os.listdir(patchs_path)
        labels = os.listdir(labels_path)

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
):
    """
    Download data from a specified source, department, and year.

    Parameters:
    - source (str): The data source identifier.
    - dep (str): The department identifier.
    - year (str): The year for which data should be downloaded.

    """

    all_exist = all(os.path.exists(f"{directory}") for directory in [patchs_path, labels_path])

    if all_exist:
        return None

    try:
        # Initialize S3 file system
        fs = get_file_system()

        print("\n*** Téléchargement des données...\n")
        # Download data from S3 to local path
        for path_local, path_s3 in zip(
            [patchs_path, labels_path],
            [x.replace("data/", "projet-slums-detection/") for x in [patchs_path, labels_path]],
        ):
            if not os.path.exists(path_local):
                fs.download(rpath=path_s3, lpath=path_local, recursive=True)

    except FileNotFoundError:
        print(f"Error: The specified data path '{path_s3}' does not exist on the S3 bucket.")
        raise

    except Exception as e:
        print(f"Error: An unexpected error occurred during the download process. {e}")
        raise
