#! /bin/bash
# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

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

if [ "$USE_S3" -eq 0 ]; then
    # Copy data locally
    mc cp -r s3/projet-slums-detection/data-preprocessed/patchs/$TASK/$SOURCE/$DEP/$YEAR/$TILES_SIZE data/data-preprocessed/patchs/$TASK/$SOURCE/$DEP/$YEAR/
    mc cp -r s3/projet-slums-detection/data-preprocessed/labels/$TYPE_LABELER/$TASK/$SOURCE/$DEP/$YEAR/$TILES_SIZE data/data-preprocessed/labels/$TYPE_LABELER/$TASK/$SOURCE/$DEP/$YEAR/
fi

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
    -P from_s3=$USE_S3
