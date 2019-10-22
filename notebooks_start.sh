sudo apt-get update && \
sudo apt-get install pkg-config && \
sudo jupyter nbextension enable --py widgetsnbextension && \
sudo jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
pip install -r requirements.txt && \
conda install -y r-rgdal && \
sudo apt-get install -y gdal-bin && \
conda install -y r-spdep && \
sudo apt install libspatialindex-dev && \
Rscript ./src_import/modules.R


