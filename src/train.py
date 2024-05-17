"""
Main script.
"""
from typing import List, Tuple
import argparse
import gc
import os
import numpy as np
import random

import albumentations as A
import mlflow
import torch
from albumentations.pytorch.transforms import ToTensorV2
from osgeo import gdal
from torch import Generator
from torch.utils.data import DataLoader, random_split

from functions.download_data import (
    get_patchs_labels,
    normalization_params,
    get_golden_paths,
    pooled_std_dev,
)
from functions.instanciators import get_dataset, get_lightning_module, get_trainer
from functions.filter import filter_indices_from_labels

gdal.UseExceptions()






remote_server_uri = "https://projet-slums-detection-128833.user.lab.sspcloud.fr"
experiment_name = "test-dev"
run_name: "kikito_stagios"
task = "segmentation"
source = "PLEIADES"
deps =  ["MARTINIQUE"]
years = ["2022"]
tiles_size = 250
augment_size = 512
type_labeler = "BDTOPO"
n_bands = 3
logits = 1
freeze_encoder = 0
epochs = 10
batch_size = 8
test_batch_size = 8
num_sanity_val_steps = 1
accumulate_batch = 8
module_name = "segformer-b5"
loss_name: "cross_entropy_weighted"
building_class_weight = 1
label_smoothing = 0.0
lr = 0.00005
momentum = float
scheduler_name = "one_cycle"
scheduler_patience = 3
patience = 200
from_s3 = 0
seed = 12345 
cuda = 0
cuda = cuda and torch.cuda.is_available()

# Seeds
torch.manual_seed(seed)
if cuda:
    print("hello")
    torch.cuda.manual_seed(seed)

random.seed(0)
np.random.seed(0)

kwargs = {"num_workers": os.cpu_count(), "pin_memory": True} if cuda else {}

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
    # dep, year  = deps[0],years[0]
    # Get patchs and labels for training
    patches, labels = get_patchs_labels(
        from_s3, task, source, dep, year, tiles_size, type_labeler, train=True
    )

    patches.sort()
    labels.sort()
    # No filtering here
    indices = filter_indices_from_labels(labels, -1.0, 2.0)
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

# Golden test dataset
golden_patches, golden_labels = get_golden_paths(
    from_s3, task, source, "MAYOTTE_CLEAN", "2022", tiles_size
)

golden_patches.sort()
golden_labels.sort()

# 2- Define the transforms to apply
# Normalization mean
normalization_mean = np.average(
    [mean[:n_bands] for mean in normalization_means], weights=weights, axis=0
)
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

# 3- Retrieve the Dataset object given the params
# TODO: mettre en Params comme Tom a fait dans formation-mlops
dataset = get_dataset(task, train_patches, train_labels, n_bands, from_s3, transform)
test_dataset = get_dataset(task, test_patches, test_labels, n_bands, from_s3, test_transform)
golden_dataset = get_dataset(
    task, golden_patches, golden_labels, n_bands, from_s3, test_transform
)

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
golden_loader = DataLoader(
    golden_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs
)

# 6- Create the trainer and the lightning
trainer = get_trainer(earlystop, checkpoints, epochs, num_sanity_val_steps, accumulate_batch)

light_module = get_lightning_module(
    module_name=module_name,
    loss_name=loss_name,
    building_class_weight=building_class_weight,
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
with mlflow.start_run(run_name=run_name):
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
            "normalization_mean": normalization_mean.tolist(),
            "normalization_std": normalization_std,
        }
    )
    # TODO: Add signature for inference

    # 8- Test
    trainer.test(dataloaders=[test_loader, golden_loader], ckpt_path="best")


def format_datasets(mayotte_2022: bool, martinique_2022: bool) -> Tuple[List[str], List[int]]:
"""
Format datasets.

Args:
    mayotte_2022 (bool): True if Mayotte 2022 dataset is used, False otherwise.
    martinique_2022 (bool): True if Martinique 2022 dataset is used, False otherwise.
Returns:
    Tuple[List[str], List[int]]: List of departments and years.
"""
deps = []
years = []
if mayotte_2022:
    deps.append("MAYOTTE_CLEAN")
    years.append(2022)
if martinique_2022:
    deps.append("MARTINIQUE")
    years.append(2022)
return deps, years


# Rajouter dans MLflow un fichier texte avc tous les nom des images used pour le training
# Dans le prepro check si habitation ou non et mettre dans le nom du fichier

datasets = {
    "mayotte_2022": args_dict.pop("mayotte_2022"),
    "martinique_2022": args_dict.pop("martinique_2022"),
}
deps, years = format_datasets(**datasets)
main(**args_dict, deps=deps, years=years)
