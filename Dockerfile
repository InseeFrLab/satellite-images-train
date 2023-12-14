FROM inseefrlab/onyxia-vscode-pytorch:py3.10.13

COPY requirements.txt requirements.txt

RUN mamba install -c conda-forge gdal -y &&\
    export PROJ_LIB=/opt/mamba/share/proj &&\
    pip install -r requirements.txt
