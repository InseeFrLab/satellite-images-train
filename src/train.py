"""
Main script.
"""

import argparse
import gc

import albumentations as A
import mlflow
import torch
from albumentations.pytorch.transforms import ToTensorV2
from osgeo import gdal
from torch import Generator
from torch.utils.data import DataLoader, random_split

from functions.download_data import get_patchs_labels
from functions.instanciators import get_dataset, get_lightning_module, get_trainer

gdal.UseExceptions()

# Command-line arguments
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
    "--dep",
    type=str,
    choices=["CAYENNE", "GUADELOUPE", "MARTINIQUE", "MAYOTTE", "REUNION"],
    default="MAYOTTE",
    help="Departement used for the training",
    required=True,
)
parser.add_argument(
    "--year",
    type=int,
    choices=[2017, 2018, 2019, 2020, 2021, 2022],
    metavar="N",
    default=2022,
    help="Year used for the training",
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
    "--type_labeler",
    type=str,
    choices=["BDTOPO"],
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
    "--epochs", type=int, default=10, metavar="N", help="Number of epochs to train (default: 10)"
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="Learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
)
parser.add_argument(
    "--module_name",
    type=str,
    choices=["deeplabv3"],
    default="deeplabv3",
    help="Model used as based model",
)
parser.add_argument(
    "--loss_name",
    type=str,
    choices=["crossentropy", "bce"],
    default="crossentropy",
    help="Loss used during the training process",
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
    "--scheduler_patience",
    type=int,
    default=10,
    help="Number of epochs with no improvement after which learning rate will be reduced",
)
parser.add_argument(
    "--from-s3",
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
    dep: str,
    year: str,
    tiles_size: int,
    type_labeler: str,
    n_bands: str,
    epochs: int,
    num_sanity_val_steps: int,
    accumulate_batch: int,
    module_name: str,
    loss_name: str,
    lr: float,
    momentum: float,
    scheduler_patience: int,
    from_s3: int,
    seed: int,
    cuda: int,
):
    """
    Main method.
    """

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

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
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=Generator())

        # 5- Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, **kwargs)

        # 6- Create the trainer and the lightning
        trainer = get_trainer(
            earlystop, checkpoints, epochs, num_sanity_val_steps, accumulate_batch
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
            cuda,
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
    main(**vars(args))
