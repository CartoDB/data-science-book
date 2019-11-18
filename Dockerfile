FROM jupyter/datascience-notebook

USER root
RUN apt-get update
RUN apt-get install build-essential software-properties-common -y
RUN apt-get install pkg-config
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN apt-get install -y gdal-bin
RUN apt-get install -y libspatialindex-dev

USER jovyan
COPY . /tmp/
RUN pip install --requirement /tmp/requirements.txt
RUN conda install -y r-rgdal
RUN conda install -y r-spdep
RUN Rscript /tmp/src_import/modules.R

