#!/bin/bash
git config --global credential.helper store

AWS_ACCESS_KEY_ID=`vault kv get -field=ACCESS_KEY_ID onyxia-kv/projet-slums-detection/s3` && export AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=`vault kv get -field=SECRET_ACCESS_KEY onyxia-kv/projet-slums-detection/s3` && export AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

sudo apt-get update
sudo apt-get install python3-gdal -y
sudo apt-get install libgdal-dev -y
pip install -r requirements.txt
pre-commit install
