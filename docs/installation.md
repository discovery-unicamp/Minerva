# Installation

Minerva is available at [PyPI](https://pypi.org/project/minerva/) for production use, and at [GitHub](https://github.com/discovery-unicamp/Minerva.git) for development.

We recommend using [uv](https://docs.astral.sh/uv/) for all installation workflows. uv is a fast Python package and project manager that handles virtual environments, dependency locking, and tool execution in one tool. pip works as a drop-in alternative everywhere uv is shown.

If you don't have uv installed, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Install via PyPI

**With uv (recommended):**

```bash
uv pip install minerva
```

**With pip:**

```bash
pip install minerva
```

Both commands install the latest version of Minerva and all its dependencies.


## Install Locally

Installing Minerva in development mode lets you work on the codebase and test changes without reinstalling after each edit. This is the recommended setup for contributors.

1. Clone the repository:

```bash
git clone https://github.com/discovery-unicamp/Minerva.git
cd Minerva
```

2. Install with development dependencies.

**With uv (recommended)** — creates a virtual environment and installs everything from the lock file:

```bash
uv sync --extra dev
```

**With pip:**

```bash
pip install -e ".[dev]"
```

Or, you can create a conda environment and install Minerva in it:

```bash
conda env create -f environment.yaml
conda activate minerva-dev
```



## Using Minerva DevContainer for developing with Minerva

Using the Minerva DevContainer is the recommended way to develop with Minerva. It provides a consistent development environment for all developers and ensures that all dependencies are installed and configured correctly.

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Visual Studio Code](https://code.visualstudio.com/)
- [VSCode Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

#### For GPU support

1. If you intent to use GPU resources, first ensure you have NVIDIA drivers installed on your system. Check if `nvidia-smi` works to verify your GPU setup.
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to enable GPU support in Docker.
3. After installing the toolkit, restart Docker. Then, test if GPU support is enabled by running

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

If you see the output of `nvidia-smi`, you have successfully enabled GPU support in Docker.

### Getting Started

1. Clone the Minerva repository

```bash
git clone https://github.com/discovery-unicamp/Minerva.git
```

2. Open VSCode and open the cloned repository folder (Minerva) as your workspace.

3. Once the workspace is open, use the command Palette (`Ctrl+Shift+P`, in Linux or `Cmd+Shift+P`, in macOS) and select `Dev Container: Rebuild and Reopen in Container`.

4. The DevContainer will start building. This may take a few minutes the first time you run it. Next time you open the workspace, it will be much faster.

5. Every time the container is build the `post_start.sh` script will be executed. This script will install the project dependencies and configure the environment.

6. After the container is built, you will be inside the container. You can now start developing with Minerva.

> **Note**: If you want a dev container with conda support, you must change the `.devcontainer/devcontainer.json` file to use the `Dockerfile.conda` instead of the default `Dockerfile`, as well as the `post_start_conda.sh` script instead of the default `post_start.sh` script. This will install Minerva in a conda environment inside the container.


## Testing

Once you have Minerva installed, you can use it as any other Python package:

```python
import minerva
```

Run the unit tests with:

**With uv:**

```bash
uv run pytest tests/
```

**With pip (after activating your environment):**

```bash
pytest tests/
```


## What's Next?

- If you are new to Minerva, check the [getting started guide](getting_started.md).
- If you want to contribute to Minerva, check the [contribution guide](contributing.md).
- If you have any questions or need help, feel free to open an issue in the [GitHub repository](https://github.com/discovery-unicamp/Minerva/issues)

