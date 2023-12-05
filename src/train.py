"""
Main script.
"""

import sys

import mlflow


def main(remote_server_uri, experiment_name, run_name):
    """
    Main method.
    """

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.autolog()
        pass

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
