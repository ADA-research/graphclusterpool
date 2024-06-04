# Graph Cluster Pooling

This repository is the code supplied with our paper submission for the 21st Workshop on Mining and Learning with Graphs (MLG) at ECML-PKDD 2024.

The layer is intended to be submitted to Pytorch Geometric library (PyG) seperatly after acceptence.

## Installation

Install all required packages using the requirements.yml as:
```conda env create -f environment.yml```


## Running experiments

The repository does not include all the Datasets as they are to large. The user is expected to download them from ``https://chrsmrrs.github.io/datasets/``. They must then be converted to the pytorch tensor format, as shown as an example in ``Datasets/PROTEINS/PROTEINS_full/convert-protein.py``.

To run an experiment, make sure to run the code on a server with Slurm available. Then simply pick an experiment (data set) to rerun. For example, PROTEIN:
```run-sbatch.sh batch-protein.sh```

## Using the layer

Although we aim to release the layer on PyG, in the mean time the layer can be used by simply importing the ```cluster_pool.py``` file into your code, which contains the ```ClusterPooling``` class. This class can be used in the same way as many other pooling layers.


## Other
The ```other``` dir contains some temporary code used to convert certain formats to create some needed information for plotting graphs etc. It is included for the advanced reader, as it is undocumented.
