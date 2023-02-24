# Becoming a Spatial Data Scientist Materials

Example notebooks to accompany [Becoming a Spatial Data Scientist](https://go.carto.com/ebooks/spatial-data-science).

![](https://go.carto.com/hubfs/spatial-data-scientist-ebook-cover.png)



## Installation requirements

The notebooks in this repository use a ready-to-run Docker image containing Jupyter applications and interactive computing tools. To run the notebooks, please follow the instructions below.

1. Clone this repository 
  ```bash
  $ git clone https://github.com/CartoDB/data-science-book.git
  $ cd data-science-book
  ```
  
2. Download and install docker. Follow instructions here: https://docs.docker.com/install/

3. Run the image. Open your terminal and run 
  ```bash
  $ docker run --user root -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -e GRANT_SUDO=yes -v "$PWD":/home/jovyan/workspace cartodb/data-science-book
  ```

  A local address will be created. Copy and paste the address in your browser, this will launch Jupyter Lab. **Note**: If you have another Jupyter server running, make sure it's on a different port than 8888. Otherwise change the port number above or close down the other notebook server. Just change ```$PWD``` to the appropriate path and you'll be good to go!

4. Start experimenting with the code in each of the Chapter directories 



## Table of Contents

### Chapter 1 - 2

- `Visualizing spatial data with CARTOframes` ([static preview](https://nbviewer.jupyter.org/github/CartoDB/data-science-book/blob/master/Chapter%201-2/Visualizing%20spatial%20data%20with%20CARTOframes.ipynb)) - a notebook for easily visualizing your data on a map using CARTOframes.

- `Computing measures of spatial dependence` ([static preview](https://nbviewer.jupyter.org/github/CartoDB/data-science-book/blob/master/Chapter%201-2/Computing%20measures%20of%20spatial%20dependence.ipynb)) - a notebook for exploring spatial dependence in your data and visualize the results using CARTOframes.

- `Discrete spatial models` ([static preview](https://nbviewer.jupyter.org/github/CartoDB/data-science-book/blob/master/Chapter%201-2/Discrete%20Spatial%20Models.ipynb)) - a notebook with examples of spatial models for discrete processes and visualize the results using CARTOframes.

- `Continous spatial models`  ([static preview](https://nbviewer.jupyter.org/github/CartoDB/data-science-book/blob/master/Chapter%201-2/Continuous%20Spatial%20Models.ipynb)) - a notebook with examples of spatial models for continuous processes and visualize the results using CARTOframes.

### Chapter 3

- `Agglomerative Clustering` ([static preview](https://nbviewer.jupyter.org/github/CartoDB/data-science-book/blob/master/Chapter%203/agglomerative.ipynb)) - a notebook demonstrating how to create spatially constrained clusters using agglomerative clustering
- `DBSCAN` ([static preview](https://nbviewer.jupyter.org/github/CartoDB/data-science-book/blob/master/Chapter%203/dbscan.ipynb)) - a notebook demonstrating how to create clusters of points in geographic coordinates
- `SKATER` ([static preview](https://nbviewer.jupyter.org/github/CartoDB/data-science-book/blob/master/Chapter%203/skater.ipynb)) - a notebook demonstrating how to create spatially constrained clusters that are homogeneous

### Chapter 4

- `Travelling Salesman Problem` ([static preview](https://nbviewer.jupyter.org/github/CartoDB/data-science-book/blob/master/Chapter%204/Travelling%20Salesman%20Problem.ipynb)) - a notebook demonstrating how to solve travelling salesman problem.
