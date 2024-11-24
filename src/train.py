"""
Main script.
"""

import argparse
import ast
import gc
import os
import random
from typing import Tuple

import albumentations as A
import mlflow
import numpy as np
import requests
import torch
from albumentations.pytorch.transforms import ToTensorV2
from osgeo import gdal
from torch import Generator
from torch.utils.data import DataLoader, random_split

from functions.download_data import (
    get_file_system,
    get_patchs_labels,
    normalization_params,
    pooled_std_dev,
)
from functions.filter import filter_indices_from_labels
from functions.instanciators import get_dataset, get_lightning_module, get_trainer

gdal.UseExceptions()


# Command-line arguments
def str_to_list(arg):
    # Convert the argument string to a list
    return ast.literal_eval(arg)


parser = argparse.ArgumentParser(description="PyTorch Training Satellite Images")
parser.add_argument(
    "--remote_server_uri",
    type=str,
    default="https://projet-slums-detection-***.user.lab.sspcloud.fr",
    help="MLflow URI",
    required=True,
)
parser.add_argument(
    "--experiment_name",
    type=str,
    choices=["segmentation", "detection", "classification", "test"],
    default="test",
    help="Experiment name in MLflow",
)
parser.add_argument(
    "--run_name",
    type=str,
    default="default",
    help="Run name in MLflow",
)
parser.add_argument(
    "--task",
    type=str,
    choices=["segmentation", "detection", "classification"],
    default="segmentation",
    help="Task of the training",
    required=True,
)
parser.add_argument(
    "--source",
    type=str,
    choices=["PLEIADES", "SENTINEL2"],
    default="PLEIADES",
    help="Source of the data used for the training",
    required=True,
)
parser.add_argument(
    "--datasets",
    type=str_to_list,
    default="['mayotte_2019', 'mayotte_2020', 'mayotte_2021']",
    help="List of datasets to be used for training",
    required=True,
)
parser.add_argument(
    "--tiles_size",
    type=int,
    choices=[250, 125],
    metavar="N",
    default=250,
    help="Size of tiles used for the training",
    required=True,
)
parser.add_argument(
    "--augment_size",
    type=int,
    metavar="N",
    default=250,
    help="Size of input tiles after augmentation",
)
parser.add_argument(
    "--type_labeler",
    type=str,
    choices=["BDTOPO", "COSIA"],
    default="BDTOPO",
    help="Source of data used for labelling",
)
parser.add_argument(
    "--n_bands",
    type=int,
    default=3,
    metavar="N",
    help="Number of bands used for the training",
)
parser.add_argument(
    "--logits",
    type=int,
    choices=[0, 1],
    default=0,
    help="Should model outputs be logits or probabilities",
)
parser.add_argument(
    "--freeze_encoder",
    type=int,
    choices=[0, 1],
    default=0,
    help="Should the encoder be frozen",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "--test_batch_size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for testing (default: 32)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="Number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="Learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--module_name",
    type=str,
    choices=[
        "deeplabv3",
        "single_class_deeplabv3",
        "segformer-b0",
        "segformer-b1",
        "segformer-b2",
        "segformer-b3",
        "segformer-b4",
        "segformer-b5",
    ],
    default="deeplabv3",
    help="Model used as based model",
)
parser.add_argument(
    "--loss_name",
    type=str,
    choices=[
        "cross_entropy",
        "cross_entropy_ignore_0",
        "cross_entropy_weighted",
        "bce",
        "bce_logits_weighted",
    ],
    default="cross_entropy",
    help="Loss used during the training process",
)
parser.add_argument(
    "--label_smoothing",
    type=float,
    default=0.0,
    help="Label smoothing the loss function",
)
parser.add_argument(
    "--building_class_weight",
    type=float,
    default=1,
    help="Weight for the building class in the loss function",
)
parser.add_argument(
    "--num_sanity_val_steps",
    type=int,
    default=2,
    help="Number of batches of val runned before starting the training routine",
)
parser.add_argument(
    "--accumulate_batch",
    type=int,
    default=8,
    help="Number of batches used for accumlate gradient",
)
parser.add_argument(
    "--scheduler_name",
    type=str,
    choices=["reduce_on_plateau", "one_cycle"],
    default="reduce_on_plateau",
    help="Scheduling policy",
)
parser.add_argument(
    "--scheduler_patience",
    type=int,
    default=3,
    help="Number of epochs with no improvement after which learning rate will be reduced",
)
parser.add_argument(
    "--patience",
    type=int,
    default=10,
    help="Number of epochs with no improvement after which training stops",
)
parser.add_argument(
    "--from_s3",
    type=int,
    choices=[0, 1],
    default=0,
    help="Read images directly from s3",
)
parser.add_argument(
    "--cuda",
    type=int,
    choices=[0, 1],
    default=0,
    help="Enables or disables CUDA training",
)
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()


def main(
    remote_server_uri: str,
    experiment_name: str,
    run_name: str,
    task: str,
    source: str,
    deps: Tuple[str],
    years: Tuple[str],
    tiles_size: int,
    augment_size: int,
    type_labeler: str,
    n_bands: str,
    logits: int,
    freeze_encoder: int,
    epochs: int,
    batch_size: int,
    test_batch_size: int,
    num_sanity_val_steps: int,
    accumulate_batch: int,
    module_name: str,
    loss_name: str,
    building_class_weight: float,
    label_smoothing: float,
    lr: float,
    momentum: float,
    scheduler_name: str,
    scheduler_patience: int,
    patience: int,
    from_s3: int,
    seed: int,
    cuda: int,
):
    """
    Main method.
    """

    # Seeds
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    random.seed(0)
    np.random.seed(0)

    kwargs = {"num_workers": os.cpu_count(), "pin_memory": True} if args.cuda else {}

    earlystop = {"monitor": "validation_loss", "patience": patience, "mode": "min"}
    checkpoints = [
        {
            "monitor": "validation_loss",
            "save_top_k": 1,
            "save_last": False,
            "mode": "min",
        }
    ]

    train_patches = []
    train_labels = []
    test_patches = []
    test_labels = []
    normalization_means = []
    normalization_stds = []
    weights = []
    for dep, year in zip(deps, years):
        # Get patchs and labels for training
        patches, labels = get_patchs_labels(
            from_s3, task, source, dep, year, tiles_size, type_labeler, train=True
        )
        patches.sort()
        labels.sort()
        # No filtering here
        indices = filter_indices_from_labels(labels, -1.0, 2.0, type_labeler)
        train_patches += [patches[idx] for idx in indices]
        train_labels += [labels[idx] for idx in indices]

        # Get patches and labels for test
        patches, labels = get_patchs_labels(
            from_s3, task, source, dep, year, tiles_size, type_labeler, train=False
        )

        patches.sort()
        labels.sort()
        test_patches += list(patches)
        test_labels += list(labels)

        # Get normalization parameters
        normalization_mean, normalization_std = normalization_params(
            task, source, dep, year, tiles_size, type_labeler
        )
        normalization_means.append(normalization_mean)
        normalization_stds.append(normalization_std)
        weights.append(len(indices))

    # # Golden test dataset
    # golden_patches, golden_labels = get_golden_paths(
    #     from_s3, task, source, "MAYOTTE_CLEAN", "2022", tiles_size
    # )
    # golden_patches.sort()
    # golden_labels.sort()

    # 2- Define the transforms to apply
    # Normalization mean
    normalization_mean = np.average(
        [mean[:n_bands] for mean in normalization_means], weights=weights, axis=0
    ).tolist()
    normalization_std = [
        pooled_std_dev(
            weights,
            [mean[i] for mean in normalization_means],
            [std[i] for std in normalization_stds],
        )
        for i in range(n_bands)
    ]

    transform_list = [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Normalize(
            max_pixel_value=1.0,
            mean=normalization_mean,
            std=normalization_std,
        ),
        ToTensorV2(),
    ]
    if augment_size != tiles_size:
        transform_list.insert(0, A.Resize(augment_size, augment_size))
    transform = A.Compose(transform_list)
    # Test transform
    test_transform_list = [
        A.Normalize(
            max_pixel_value=1.0,
            mean=normalization_mean,
            std=normalization_std,
        ),
        ToTensorV2(),
    ]
    if augment_size != tiles_size:
        test_transform_list.insert(0, A.Resize(augment_size, augment_size))
    test_transform = A.Compose(test_transform_list)

    # train_patches = train_patches[:100]
    # train_labels = train_labels[:100]
    # test_patches = test_patches[:100]
    # test_labels = test_labels[:100]

    # 3- Retrieve the Dataset object given the params
    # TODO: mettre en Params comme Tom a fait dans formation-mlops
    dataset = get_dataset(task, train_patches, train_labels, n_bands, from_s3, transform)
    test_dataset = get_dataset(task, test_patches, test_labels, n_bands, from_s3, test_transform)
    # golden_dataset = get_dataset(
    #     task, golden_patches, golden_labels, n_bands, from_s3, test_transform
    # )

    # 4- Use random_split to split the dataset
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=Generator())

    # 5- Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs
    )
    # golden_loader = DataLoader(
    #     golden_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs
    # )

    # 6- Create the trainer and the lightning
    trainer = get_trainer(earlystop, checkpoints, epochs, num_sanity_val_steps, accumulate_batch)

    weights = [
        building_class_weight if label == "Bâtiment" else 1.0
        for label in requests.get(
            f"https://minio.lab.sspcloud.fr/projet-slums-detection/data-label/{type_labeler}/{type_labeler.lower()}-id2label.json"
        )
        .json()
        .values()
    ]

    light_module = get_lightning_module(
        module_name=module_name,
        type_labeler=type_labeler,
        loss_name=loss_name,
        weights=weights,
        label_smoothing=label_smoothing,
        n_bands=n_bands,
        logits=bool(logits),
        freeze_encoder=bool(freeze_encoder),
        task=task,
        lr=lr,
        momentum=momentum,
        earlystop=earlystop,
        scheduler_name=scheduler_name,
        scheduler_patience=scheduler_patience,
        cuda=cuda,
    )

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.pytorch.autolog()
        # 7- Training the model on the training set
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")
        gc.collect()

        trainer.fit(light_module, train_loader, val_loader)

        best_model = type(light_module).load_from_checkpoint(
            checkpoint_path=trainer.checkpoint_callback.best_model_path,
            model=light_module.model,
            loss=light_module.loss,
            optimizer=light_module.optimizer,
            optimizer_params=light_module.optimizer_params,
            scheduler=light_module.scheduler,
            scheduler_params=light_module.scheduler_params,
            scheduler_interval=light_module.scheduler_interval,
        )

        # Logging the model with the associated code
        mlflow.pytorch.log_model(
            artifact_path="model",
            code_paths=[
                "src/models/",
                "src/optim/",
                "src/config/",
            ],
            pytorch_model=best_model.to("cpu"),
        )

        # Log normalization parameters
        mlflow.log_params(
            {
                "normalization_mean": normalization_mean,
                "normalization_std": normalization_std,
            }
        )
        # TODO: Add signature for inference

        # 8- Test
        trainer.test(dataloaders=[test_loader, test_loader], ckpt_path="best")
        run_id = run.info.run_id
        return run_id


def format_datasets(args_dict: dict) -> Tuple[str, int]:
    """
    Format datasets.

    Args:
        args_dict (dict): A dictionary containing the command-line arguments.

    Returns:
        Tuple[str, int]: A tuple containing the list of departments and years extracted from the dataset names.

    Raises:
        ValueError: If the S3 path does not exist.

    """
    deps, years = zip(*[item.split("_") for item in args_dict["datasets"]])
    deps = [dep.upper() for dep in deps]
    fs = get_file_system()
    for dep, year in zip(deps, years):
        s3_path = f"s3://projet-slums-detection/data-raw/{args_dict['source']}/{dep.upper()}/{year}"
        if not fs.exists(s3_path):
            raise ValueError(f"S3 path {s3_path} does not exist.")
    args_dict.pop("datasets")
    return deps, years, args_dict


# Rajouter dans MLflow un fichier texte avc tous les nom des images used pour le training

if __name__ == "__main__":
    args_dict = vars(args)
    deps, years, args_dict = format_datasets(args_dict)
    run_id = main(**args_dict, deps=deps, years=years)
    print(run_id)
