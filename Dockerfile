FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# Ensure apt-get is in non-interactive mode during build.
ARG DEBIAN_FRONTEND=interactive

# Enforce UTF-8 encoding, needed by various components.
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

WORKDIR /tmp

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

# Install Python package dependencies.
RUN apt-get update && apt-get install --yes \
    libsm6 \
    libxrender-dev \
&&  rm -rf /var/lib/apt/lists/*

# Install required Python 3 packages via Pipenv.
RUN pip3 install --no-cache-dir pipenv
ADD Pipfile* /tmp/
RUN pipenv install --system --deploy

# Bundle executable project files.
# ADD bin/* /usr/local/bin/
# ADD python /root/python

WORKDIR /root
