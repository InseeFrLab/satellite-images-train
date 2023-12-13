#! /bin/bash

MLFLOW_TRACKING_URI="https://projet-slums-detection-128833.user.lab.sspcloud.fr"
MLFLOW_EXPERIMENT_NAME="segmentation"
ENTRY_POINT=main

TASK=segmentation
SOURCE=PLEIADES
DEP=MAYOTTE
YEAR=2022
TILES_SIZE=250
TYPE_LABELER=BDTOPO
USE_S3=0
EPOCHS=2
LR=0.0001
CUDA=1

mlflow run ~/work/satellite-images-train/ \
    --env-manager=local \
    --entry-point $ENTRY_POINT \
    -P remote_server_uri=$MLFLOW_TRACKING_URI \
    -P experiment_name=$MLFLOW_EXPERIMENT_NAME \
    -P source=$SOURCE \
    -P dep=$DEP \
    -P year=$YEAR \
    -P type_labeler=$TYPE_LABELER \
    -P task=$TASK \
    -P tiles_size=$TILES_SIZE \
    -P epochs=$EPOCHS \
    -P lr=$LR \
    -P from_s3=$USE_S3 \
    -P cuda=$CUDA
