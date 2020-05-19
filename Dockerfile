FROM tensorflow/tensorflow:2.2.0-gpu

# Ensure apt-get is in non-interactive mode during build.
ARG DEBIAN_FRONTEND=noninteractive

# Enforce UTF-8 encoding, needed by various components.
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

WORKDIR /tmp

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

# Install Python 3 development headers, etc.
RUN apt-get purge --yes --autoremove \
    python3-dev \
    python3-distutils \
    python3-lib2to3 \
    python3-pip
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install --yes \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3.8-lib2to3 \
&&  rm -rf /var/lib/apt/lists/*

# Install latest version of pip.
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.8

# Install Python package dependencies.
RUN apt-get update && apt-get install --yes \
    libsm6 \
    libxrender-dev \
&&  rm -rf /var/lib/apt/lists/*

# Install required Python 3 packages via Pipenv.
RUN pip3.8 install --no-cache-dir pipenv
ADD Pipfile* /tmp/
RUN pipenv install --system --deploy

# Install custom TensorFlow package compiled for Springfield.
ADD tensorflow-2.2.0-cp38-cp38-linux_x86_64.whl /tmp/
RUN pip3.8 install --force-reinstall tensorflow-2.2.0-cp38-cp38-linux_x86_64.whl

# Bundle executable project files.
# ADD bin/* /usr/local/bin/
# ADD python /root/python

RUN rm -rf /tmp/*

WORKDIR /root
