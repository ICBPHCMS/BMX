# BMX
B discriMinant eXtreme

![status](https://api.travis-ci.org/ICBPHCMS/BMX.svg?branch=master)

## Setup the environment
The following steps have to be done only once. It will setup an [anaconda](https://www.anaconda.com/) and install the required software packages. These are amongst others [keras v2.1.5](https://keras.io/) and [tensorflow v1.6](https://www.tensorflow.org/).
* A single environment for CPU only: ```source Env/setupEnvCPU.sh  <install_dir>```
* Two environents for CPU and GPU (recommended): ```source Env/setupEnvFull.sh  <install_dir>```
After the enviroment is setup it can be used with ```source Env/env_cpu.sh``` or ```source Env/env_gpu.sh``` respectively.

## Perparing training data
The script ```Training/unpackForTraining.py``` reads extended nanoaod files, extracts and transforms the input observables per B hypothesis, randomizes signal (from MC) and background (from sideband region in data), and splits events into training and testing sets. The output is saved as hdf5 files.

## Training
Training on the prepared train/test hdf5 files is done with the script ```Training/training.py```. To perform the training on a GPU an installation of [cuda v9.0](https://developer.nvidia.com/cuda-90-download-archive) and [cudnn v7 for cuda9](https://developer.nvidia.com/cudnn) as well as supporting nvidia graphics drivers have to be install on the corresponding node.
