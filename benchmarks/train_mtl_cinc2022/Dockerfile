# FROM python:3.8.6-slim
# https://hub.docker.com/r/nvidia/cuda/
# FROM nvidia/cuda:11.1.1-devel
# FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
# NOTE: the base image has python version 3.7

# NOTE: The GPU provided by the Challenge is GPU Tesla T4 with nvidiaDriverVersion: 470.82.01
# by checking https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# and https://download.pytorch.org/whl/torch_stable.html


## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

# submodule
# RUN apt-get update && \
#     apt-get upgrade -y && \
#     apt-get install -y git

## Install your dependencies here using apt install, etc.

# latest version of biosppy uses opencv
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt update
RUN apt install build-essential -y
RUN apt install git ffmpeg libsm6 libxext6 vim libsndfile1 -y

# RUN apt update && apt install -y --no-install-recommends \
#         build-essential \
#         curl \
#         software-properties-common \
#         unzip

RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# RUN pip install --upgrade pip

# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
# RUN pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# compatible with torch
RUN pip install torchaudio==0.10.0+cu113 --no-deps -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torch
# RUN pip install git+https://github.com/DeepPSP/torch_ecg.git
# RUN pip install git+https://github.com/asteroid-team/torch-audiomentations.git --no-deps
RUN pip install torch-ecg
RUN pip install torch-audiomentations --no-deps


# NOTE: also run test_local.py to test locally
# since GitHub Actions does not have GPU,
# one need to run test_local.py to avoid errors related to devices
RUN python test_docker.py


# commands to run test with docker container:

# sudo docker build -t image .
# sudo docker run -it --shm-size=10240m --gpus all -v ~/Jupyter/temp/cinc2022_docker_test/model:/physionet/model -v ~/Jupyter/temp/cinc2022_docker_test/test_data:/physionet/test_data -v ~/Jupyter/temp/cinc2022_docker_test/test_outputs:/physionet/test_outputs -v ~/Jupyter/temp/cinc2022_docker_test/data:/physionet/training_data image bash


# python train_model.py training_data model
# python test_model.py model test_data test_outputs
