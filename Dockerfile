FROM inseefrlab/onyxia-vscode-pytorch:py3.10.13

ENV PROJ_LIB=/opt/mamba/share/proj

COPY requirements.txt requirements.txt

RUN mamba install -c conda-forge gdal -y &&\
    pip install -r requirements.txt
