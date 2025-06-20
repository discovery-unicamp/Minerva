FROM python:3.10-slim-bullseye

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install needed apt packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl ffmpeg git ssh unzip wget sudo && \
    rm -rf /var/lib/apt/lists/* && ldconfig && apt autoremove -y && apt clean

# Create user and setup sudo
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && chmod 0440 /etc/sudoers.d/$USERNAME && \
    groupmod --gid $USER_GID $USERNAME && usermod --uid $USER_UID --gid $USER_GID $USERNAME && \
    chown -R $USER_UID:$USER_GID /home/$USERNAME

# Add ~/.local/bin to PATH in .bashrc
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/$USERNAME/.bashrc

# Change default shell to bash
RUN chsh --shell /bin/bash $USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME
