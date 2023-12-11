"""
Main script.
"""

import gc
import sys
from typing import Dict, List

import albumentations as A
import mlflow
import torch
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

        # 1- Get patch paths from s3
        patchs = fs.ls(
            (
                f"projet-slums-detection/data-preprocessed/patchs/"
                f"{task}/{source}/{dep}/{year}/{tiles_size}"
            )
        )

        # 2- Get label paths from s3
        labels = fs.ls(
            (
                f"projet-slums-detection/data-preprocessed/labels/"
                f"{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}"
            )
        )

        # 3- Define the transforms to apply
        transform = A.Compose(
            [
                A.RandomResizedCrop(*(tiles_size,) * 2, scale=(0.7, 1.0), ratio=(0.7, 1)),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                # TODO: Calculer moyenne et variance sur toutes les images
                A.Normalize(
                    max_pixel_value=255.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ]
        )

        # 4- Retrieve the Dataset object given the params
        # TODO: mettre en Params comme Tom a fait dans formation-mlops
        dataset = get_dataset(task, patchs, labels, n_bands, fs, transform)

        # 5- Use random_split to split the dataset
        generator = Generator().manual_seed(2023)
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)

        # 6- Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # 7- Create the trainer and the lightning
        trainer = get_trainer(
            earlystop, checkpoints, max_epochs, num_sanity_val_steps, accumulate_batch
        )

        light_module = get_lightning_module(
            module_name,
            loss_name,
            n_channel,
            task,
            lr,
            momentum,
            earlystop,
            scheduler_patience,
        )

        # 8- Training the model on the training set
        torch.cuda.empty_cache()
        gc.collect()

        trainer.fit(light_module, train_loader, val_loader)

        # 9- Inference on the validation set
        # light_module.eval()
        # with torch.no_grad():
        #     for inputs, targets in val_loader:
        #         outputs = light_module(inputs)

        return trainer


# Rajouter dans MLflow un fichier texte avc tous les nom des images used pour le training
# Dans le prepro check si habitation ou non et mettre dans le nom du fichier


if __name__ == "__main__":
    main(
        str(sys.argv[1]),
        str(sys.argv[2]),
        str(sys.argv[3]),
        str(sys.argv[4]),
        str(sys.argv[5]),
        str(sys.argv[6]),
        str(sys.argv[7]),
        str(sys.argv[8]),
        str(sys.argv[9]),
        str(sys.argv[10]),
        str(sys.argv[11]),
        str(sys.argv[12]),
        str(sys.argv[13]),
        str(sys.argv[14]),
        str(sys.argv[15]),
        str(sys.argv[16]),
        str(sys.argv[17]),
        str(sys.argv[18]),
        str(sys.argv[19]),
        str(sys.argv[20]),
        str(sys.argv[21]),
    )
