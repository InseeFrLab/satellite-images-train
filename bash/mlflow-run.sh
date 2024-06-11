#! /bin/bash

export MLFLOW_TRACKING_URI=https://projet-slums-detection-128833.user.lab.sspcloud.fr
export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
export MLFLOW_EXPERIMENT_NAME=segmentation
ENTRY_POINT=main

TASK=segmentation

SOURCE=PLEIADES
DATASETS="""["MAYOTTE_2022", "MARTINIQUE_2022"]"""
TILES_SIZE=250
TYPE_LABELER=BDTOPO
USE_S3=0
EPOCHS=10
BATCH_SIZE=32
TEST_BATCH_SIZE=32
LR=0.00005
SCHEDULER_NAME=one_cycle
BUILDING_CLASS_WEIGHT=40
FREEZE_ENCODER=1
LOSS_NAME=bce_logits_weighted
MODULE_NAME=single_class_deeplabv3
LABEL_SMOOTHING=0.0
LOGITS=1
CUDA=0

mlflow run ~/work/satellite-images-train/ \
    --env-manager=local \
    --entry-point $ENTRY_POINT \
    -P remote_server_uri=$MLFLOW_TRACKING_URI \
    -P experiment_name=$MLFLOW_EXPERIMENT_NAME \
    -P source=$SOURCE \
    -P datasets=$DATASETS \
    -P type_labeler=$TYPE_LABELER \
    -P task=$TASK \
    -P tiles_size=$TILES_SIZE \
    -P epochs=$EPOCHS \
    -P batch_size=$BATCH_SIZE \
    -P test_batch_size=$TEST_BATCH_SIZE \
    -P lr=$LR \
    -P scheduler_name=$SCHEDULER_NAME \
    -P from_s3=$USE_S3 \
    -P loss_name=$LOSS_NAME \
    -P module_name=$MODULE_NAME \
    -P label_smoothing=$LABEL_SMOOTHING \
    -P logits=$LOGITS \
    -P building_class_weight=$BUILDING_CLASS_WEIGHT \
    -P freeze_encoder=$FREEZE_ENCODER \
    -P cuda=$CUDA
