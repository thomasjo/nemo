# FROM tensorflow/tensorflow:2.3.0-gpu
FROM nvidia/cuda:10.2-cudnn7-runtime

# Ensure apt-get is in non-interactive mode during build.
ARG DEBIAN_FRONTEND=noninteractive

# Enforce UTF-8 encoding, needed by various components.
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Ensure all base packages are up-to-date.
# RUN apt-get update && apt-get upgrade --yes \
# &&  rm -rf /var/lib/apt/lists/*

# Install basic tools for a sensible workflow.
RUN apt-get update && apt-get install --yes \
    build-essential \
    ca-certificates \
    curl \
    git \
    software-properties-common \
&&  rm -rf /var/lib/apt/lists/*

# Install up-to-date Python 3 with development headers, etc.
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install --yes \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    python3.7-lib2to3 \
&&  rm -rf /var/lib/apt/lists/*

# Install latest version of pip.
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.7

# Install custom TensorFlow package compiled for Springfield.
# ADD tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl /tmp/
# RUN pip3.8 install --force-reinstall tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl

# Install required Python packages.
ADD requirements.txt /tmp
RUN pip3.7 install -r /tmp/requirements.txt

RUN rm -rf /tmp/*
