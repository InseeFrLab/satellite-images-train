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

from functions.download_data import get_patchs_labels
from functions.instanciators import get_dataset, get_lightning_module, get_trainer

# source = "PLEIADES"
# dep = "GUADELOUPE"
# year = "2020"
# n_bands = 3
# type_labeler = "BDTOPO"
# task = "segmentation"
# tiles_size = 250
# experiment_name = "default"
# earlystop = {"monitor": "validation_IOU", "patience": 35, "mode": "max"}
# checkpoints = [{"monitor": "validation_IOU", "save_top_k": 1, "save_last": True, "mode": "max"}]
# max_epochs = 2
# num_sanity_val_steps = 2
# accumulate_batch = 8
# module_name = "deeplabv3"
# loss_name = "crossentropy"
# lr = 0.0001
# momentum = 0.9
# scheduler_patience = 10
# from_s3 = False


def main(
    remote_server_uri: str,
    experiment_name: str,
    run_name: str,
    task: str,
    source: str,
    dep: str,
    year: str,
    tiles_size: int,
    type_labeler: str,
    n_bands: str,
    earlystop: Dict,
    checkpoints: List[Dict],
    max_epochs: int,
    num_sanity_val_steps: int,
    accumulate_batch: int,
    module_name: str,
    loss_name: str,
    lr: float,
    momentum: float,
    scheduler_patience: int,
    from_s3: bool,
):
    """
    Main method.
    """

    earlystop = {"monitor": "validation_IOU", "patience": 35, "mode": "max"}
    checkpoints = [{"monitor": "validation_IOU", "save_top_k": 1, "save_last": True, "mode": "max"}]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.autolog()

        # Get patchs and labels
        patchs, labels = get_patchs_labels(
            from_s3, task, source, dep, year, tiles_size, type_labeler
        )

        # 2- Define the transforms to apply
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

        # 3- Retrieve the Dataset object given the params
        # TODO: mettre en Params comme Tom a fait dans formation-mlops
        dataset = get_dataset(task, patchs, labels, n_bands, from_s3, transform)

        # 4- Use random_split to split the dataset
        generator = Generator().manual_seed(2023)
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)

        # 5- Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=5)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=5)

        # 6- Create the trainer and the lightning
        trainer = get_trainer(
            earlystop, checkpoints, max_epochs, num_sanity_val_steps, accumulate_batch
        )

        light_module = get_lightning_module(
            module_name,
            loss_name,
            n_bands,
            task,
            lr,
            momentum,
            earlystop,
            scheduler_patience,
        )

        # 7- Training the model on the training set
        torch.cuda.empty_cache()
        gc.collect()

        trainer.fit(light_module, train_loader, val_loader)

        # 8- Inference on the validation set
        # light_module.eval()
        # with torch.no_grad():
        #     for inputs, targets in val_loader:
        #         outputs = light_module(inputs)

        return trainer


# Rajouter dans MLflow un fichier texte avc tous les nom des images used pour le training
# Dans le prepro check si habitation ou non et mettre dans le nom du fichier


if __name__ == "__main__":
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    print(sys.argv[4])
    print(sys.argv[5])
    print(sys.argv[6])
    print(sys.argv[7])
    print(sys.argv[8])
    print(sys.argv[9])
    print(sys.argv[10])
    print(sys.argv[11])
    print(sys.argv[12])
    print(sys.argv[13])
    print(sys.argv[14])
    print(sys.argv[15])
    print(sys.argv[16])
    print(sys.argv[17])
    print(sys.argv[18])
    print(sys.argv[19])
    print(sys.argv[20])
    print(sys.argv[21])

    main(
        str(sys.argv[1]),
        str(sys.argv[2]),
        str(sys.argv[3]),
        str(sys.argv[4]),
        str(sys.argv[5]),
        str(sys.argv[6]),
        str(sys.argv[7]),
        int(sys.argv[8]),
        str(sys.argv[9]),
        str(sys.argv[10]),
        str(sys.argv[11]),
        str(sys.argv[12]),
        int(sys.argv[13]),
        int(sys.argv[14]),
        int(sys.argv[15]),
        str(sys.argv[16]),
        str(sys.argv[17]),
        float(sys.argv[18]),
        float(sys.argv[19]),
        int(sys.argv[20]),
        bool(int(sys.argv[21])),
    )
