FROM nvcr.io/nvidia/pytorch:23.06-py3

# Install common apt packages
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update &&  \
    apt install -y  build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    curl \
    git \
    htop \
    libssl-dev \
    nvtop \
    net-tools \
    pandoc \
    python3-sphinx \
    tmux \
    tree \
    unrar \
    unzip \
    vim \
    wget \
    xdot && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig && \
    apt autoremove && \
    apt clean

# ***************** Specific packages ********************

# System-wide python packages
RUN python -m pip install --upgrade pip setuptools pytest && \
    python -m pip install lightning \
    numpy \
    pandas \
    plotly \
    PyYAML \ 
    scipy \
    statsmodels \
    tifffile \
    torch \
    zarr \
    rich \ 
    perlin-noise \
    torchmetrics 

#this will only work after the repo is public
# RUN git clone https://github.com/discovery-unicamp/Minerva
# RUN cd Minerva && python -m pip install .