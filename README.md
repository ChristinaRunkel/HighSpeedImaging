# 'A Bit Too Much? High Speed Imaging from Sparse Photon Counts' Python Implementation

This repository is a python implementation of the scheme proposed in 
```
Chandramouli, Paramanand, et al. "A bit too much? high speed imaging from sparse photon counts." 
2019 IEEE International Conference on Computational Photography (ICCP). IEEE, 2019.
```
A preprint can be found at https://arxiv.org/abs/1811.02396.

## Code
The code is structered into data loading (`dataloader.py`, `torch_augment.py`, `torch_augment_functions.py`), definition of the network architecture (`network.py`), specification of the loss function (`loss.py`) and auxiliary files to define the training and testing procedure (`solver.py`, `scheduler.py`) as well as code for n-dimensional stitching (`torch_stitching.py`). Some exemplary network snapshots can be found in `/snapshots`.
## Requirements
- PyTorch
- Scikit-image
- H5py

The exact setup can be installed by running
```
conda env create -f environment.yml
conda activate HighSpeedImaging
```
## Data
The dataset used for training and testing of the network is the [Deep Video Deblurring for Hand-held Cameras Dataset](https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/#dataset) which is publicly available.


This code has been developed together with [mj9](https://github.com/mj9) as part of a university project.