# Graph Cluster Pooling

This repository is the code supplied with our paper submission for the 21st Workshop on Mining and Learning with Graphs (MLG) at ECML-PKDD 2024.

The layer is intended to be submitted to Pytorch Geometric library (PyG) seperatly after acceptence.

## Installation

Install all required packages using the requirements.yml as:
```conda env create -f environment.yml```


## Running experiments

To run an experiment, make sure to run the code on a server with Slurm available. Then simply pick an experiment (data set) to rerun. For example, PROTEIN:
```run-sbatch.sh batch-protein.sh```

## Using the layer

Although we aim to release the layer on PyG, in the mean time the layer can be used by simply importing the ```cluster_pool.py``` file into your code, which contains the ```ClusterPooling``` class. This class can be used in the same way as many other pooling layers.