#! /bin/bash
# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

# Set MLFLOW_TRACKING_URI environment variable
export MLFLOW_TRACKING_URI="https://projet-slums-detection-128833.user.lab.sspcloud.fr"

# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_EXPERIMENT_NAME="segmentation"

# Set ENTRY_POINT
export ENTRY_POINT="segmentation"

mlflow run ~/work/satellite-images-train/ \
--env-manager=local \
--entry-point $ENTRY_POINT \
-P remote_server_uri=$MLFLOW_TRACKING_URI \
-P experiment_name=$MLFLOW_EXPERIMENT_NAME \
-P source="PLEIADES" \
-P dep="MAYOTTE" \
-P year="2022" \
-P n_bands=3 \
-P type_labeler="BDTOPO" \
-P task="segmentation" \
-P tiles_size=250 \
-P earlystop="temp" \
-P checkpoints="temp" \
-P max_epochs=2 \
-P num_sanity_val_steps=2 \
-P accumulate_batch=8 \
-P module_name="deeplabv3" \
-P loss_name="crossentropy" \
-P lr=0.0001 \
-P momentum=0.9 \
-P scheduler_patience=10 \
-P from_s3=False
