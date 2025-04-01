# Let's pull from nvidia pytorch 23.06 (https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-06.html)
# This is a ubuntu 22.04 based with:
#   - python 3.10 
#   - CUDA 12.1.1
#   - CUDNN 8.9.2
#   - Pytorch 2.1.0a0+4136153 (may 2, 2023 - https://github.com/pytorch/pytorch/tree/41361538a978eb03fa1e88bf5b8e4410db7a6927)
# It requeres Nvidia Driver release 530 or later 
FROM nvcr.io/nvidia/pytorch:23.06-py3

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# ******************************************************** 
# Install common apt packages for developing
# ********************************************************
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update &&  \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131
    apt-get purge -y imagemagick imagemagick-6-common  && \ 
    apt install -y  build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        htop \
        libsm6 \
        libssl-dev \
        libxext6 \
        nvtop \
        pandoc \
        python3-opencv \
        python3-pip \
        python3-sphinx \
        tmux \
        unrar \
        unzip \
        vim \
        wget 


# ********************************************************
# * Some common python packages
# ********************************************************
RUN python3 -m pip install --upgrade pip>=23 setuptools>=75.8.0


# ********************************************************
# Create the vscode user (non-root user) and add sudo support
# ********************************************************
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME  && \
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Change group to same as system
RUN groupmod --gid $USER_GID $USERNAME \
    && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME


# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
RUN rm -rf /var/lib/apt/lists/* && \
    ldconfig && \
    apt autoremove && \
    apt clean


# ********************************************************
# Final operations
# ********************************************************
# Change shell for user to bash
RUN chsh --shell /bin/bash $USERNAME
#  Set the default user
USER $USERNAME
