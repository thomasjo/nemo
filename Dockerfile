FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# Ensure apt-get is in noninteractive mode during build.
ARG DEBIAN_FRONTEND=interactive

# Ensure all base packages are up-to-date.
RUN apt-get update && apt-get upgrade --yes \
&&  rm -rf /var/lib/apt/lists/*

# Install basic tools for a sensible workflow.
RUN apt-get update && apt-get install --yes \
    build-essential \
    ca-certificates \
    curl \
    git \
    software-properties-common \
&&  rm -rf /var/lib/apt/lists/*

# Install Python 3 development headers, etc.
RUN apt-get update && apt-get install --yes \
    python3 \
    python3-dev \
&&  rm -rf /var/lib/apt/lists/*

# Install latest version of pip.
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

# Install required Python 3 packages.
RUN pip3 install --no-cache-dir \
    docopt \
    h5py \
    imageio \
    matplotlib \
    numpy \
    pandas \
    pyyaml \
    scikit-image \
    scikit-learn \
    scipy

# Install OpenCV with Python bindings.
RUN apt-get update && apt-get install --yes \
    python3-opencv \
&&  rm -rf /var/lib/apt/lists/*

# Install TensorFlow packages.
RUN pip3 install --no-cache-dir \
    tensorflow-hub \
    tensorflow-gpu==2.0.0a0

ADD bin/* /usr/local/bin/
ADD python /root/python
