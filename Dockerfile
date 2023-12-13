FROM inseefrlab/onyxia-vscode-pytorch:py3.10.13

COPY requirements.txt requirements.txt

RUN mamba install -c conda-forge gdal -y &&\
    pip install -r requirements.txt
