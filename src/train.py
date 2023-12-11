"""
Main script.
"""

import sys
from typing import Dict, List

import albumentations as A
import mlflow
from albumentations.pytorch.transforms import ToTensorV2
from torch import Generator
from torch.utils.data import DataLoader, random_split

from functions.download_data import get_file_system
from functions.instanciators import get_dataset, get_lightning_module, get_trainer

source = "PLEIADES"
dep = "GUADELOUPE"
year = "2020"
n_bands = 3
type_labeler = "BDTOPO"
task = "segmentation"
tiles_size = 250


def main(
    remote_server_uri: str,
    experiment_name: str,
    run_name: str,
    task: str,
    source: str,
    dep: str,
    year: str,
    tiles_size: str,
    type_labeler: str,
    n_bands: str,
    earlystop: Dict,
    checkpoints: List[Dict],
    max_epochs: int,
    num_sanity_val_steps: int,
    accumulate_batch: int,
    module_name: str,
    loss_name: str,
    n_channel: int,
    lr: float,
    momentum: float,
    scheduler_patience: int,
):
    """
    Main method.
    """

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.autolog()
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

        transform = A.Compose(
            [
                A.RandomResizedCrop(*(tiles_size,) * 2, scale=(0.7, 1.0), ratio=(0.7, 1)),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                # Calculer moyenne et variance sur toutes
                A.Normalize(
                    max_pixel_value=255.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ]
        )

        # mettre en Params comme Tom a fait dans formation-mlops
        dataset = get_dataset(task, patchs, labels, n_bands, fs, transform)

        # Use random_split to split the dataset
        generator = Generator().manual_seed(2023)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [0.7, 0.2, 0.1], generator=generator
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        trainer = get_trainer(
            earlystop, checkpoints, max_epochs, num_sanity_val_steps, accumulate_batch
        )

        lightning_module = get_lightning_module(
            module_name,
            loss_name,
            n_channel,
            task,
            lr,
            momentum,
            earlystop,
            scheduler_patience,
        )

        return train_loader, test_loader, val_loader, trainer, lightning_module

    # 1- Download data ? Est ce qu'on peut donner des path s3 au dataloader ?
    # 2- Prepare data (filtrer certaines images sans maison ? balancing)
    # 3- Split data train/test/valid => instancie dataloader
    # 4- On instancie le trainer
    # 5- On instancie le lightning_module
    # 6- On entraine le modele
    # 7- On evalue le modele
    # 8- On auto log sur MLflow


# Rajouter dans MLflow un fichier texte avc tous les nom des mages used pour le training
# Dans le prepro check si habitation ou non et mettre dans le nom du fichier


if __name__ == "__main__":
    main(
        str(sys.argv[1]),
        str(sys.argv[2]),
        str(sys.argv[3]),
    )
