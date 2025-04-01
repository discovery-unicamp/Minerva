# Using Minerva DevContainer for developing with Minerva

Using the Minerva DevContainer is the recommended way to develop with Minerva. It provides a consistent development environment for all developers and ensures that all dependencies are installed and configured correctly.

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Visual Studio Code](https://code.visualstudio.com/)
- [VSCode Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### For GPU support

1. If you intent to use GPU resources, first ensure you have NVIDIA drivers installed on your system. Check if `nvidia-smi` works to verify your GPU setup.
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to enable GPU support in Docker.
3. After installing the toolkit, restart Docker. Then, test if GPU support is enabled by running

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

If you see the output of `nvidia-smi`, you have successfully enabled GPU support in Docker.

## Getting Started

1. Clone the Minerva repository

```bash
git clone https://github.com/discovery-unicamp/Minerva-Dev.git
```

2. Open VSCode and open the cloned repository folder (Minerva-Dev) as your workspace.

3. Once the workspace is open, use the command Palette (`Ctrl+Shift+P`, in Linux or `Cmd+Shift+P`, in macOS) and select `Dev Container: Rebuild and Reopen in Container`.

4. The DevContainer will start building. This may take a few minutes the first time you run it. Next time you open the workspace, it will be much faster.

5. Every time the container is build the `post_start.sh` script will be executed. This script will install the project dependencies and configure the environment.

6. After the container is built, you will be inside the container. You can now start developing with Minerva.
