# Minerva

[![Continuous Test](https://github.com/discovery-unicamp/Minerva-Dev/actions/workflows/continuous-testing.yml/badge.svg)](https://github.com/discovery-unicamp/Minerva/actions/workflows/python-app.yml)

[![Release to PyPI](https://github.com/discovery-unicamp/Minerva/actions/workflows/release_to_pypi.yml/badge.svg)](https://github.com/discovery-unicamp/Minerva/actions/workflows/release_to_pypi.yml)

Minerva is a Pytorch-Lightning-based framework for training machine learning models for researchers.

## Description

This project aims to provide a robust and flexible framework for researchers working on machine learning projects. It includes various utilities and modules for data transformation, model creation, analysis metrics, and reproducibility. Minerva is designed to be modular and extensible, allowing researchers to easily add new features and functionalities.

### Features

Minerva offers a wide range of features to help you with your machine learning projects:

- **Model Creation**: Minerva offers a variety of models and architectures to choose from.
- **Training and Evaluation**: Minerva provides tools to train and evaluate your models, including loss functions, optimizers, and evaluation metrics.
- **Data Transformation**: Minerva provides tools to preprocess and transform your data, including data loaders, data augmentation, and data normalization.
- **Modular Design**: Minerva is designed to be modular and extensible, allowing you to easily add new features and functionalities.
- **Reproducibility**: Minerva ensures reproducibility by providing tools for versioning, configuration, and logging of experiments.
- **Self-Supervised Learning (SSL) Support**: Minerva supports Self-Supervised Learning (SSL) for training models with limited labeled data.
- **Development Environment**: Minerva provides a development environment with all dependencies pre-installed and configured for you.

### Near Future Features

- **Hyperparameter Optimization**: Minerva will offer tools for hyperparameter optimization powered by Ray Tune.
- **PyPI Package**: Minerva will be available as a PyPI package for easy installation.
- **Pre-trained Models**: Minerva will offer pre-trained models for common tasks and datasets.
- **Experiment Management**: Minerva will offer tools for managing and tracking experiments using well-known tools like MLflow.

## Installation

Minerva is currently under development and not yet available as a PyPI package. You can install it:
- Locally, as any other Python package.
- Using a Docker container, if you want to use the development environment.

### Install With pip
```bash
pip install minerva-ml
```

### Install Locally

1. Clone the repository:

```bash
git clone https://github.com/discovery-unicamp/Minerva.git
```

2. And then navigate to the project directory and install the dependencies:

```bash
cd Minerva
pip install .
```


### VSCode Development Environment using DevContainer

Check the [Using Minerva DevContainer for developing with Minerva](.devcontainer/README.md) guide for instructions on how to set up a development environment using Visual Studio Code and Docker.

## Modules

Once installed, Minerva is just like any other Python package. You can import its modules and classes in your Python code using:

```python
import minerva
```

The main modules available in Minerva are:
- `minerva.analysis`: Tools for analyzing and visualizing model performance.
- `minerva.callback`: Callbacks for monitoring and logging training progress (Pytorch-Lightning based).
- `minerva.data`: Readers, Datasets and Data Modules.
- `minerva.losses`: Loss functions.
- `minerva.models`: Models and Architectures for supervised and self-supervised learning. Check out the available models and SSL techniques at [minerva.models README.md file](minerva/models/README.md).
- `minerva.pipeline`: Tools for training and evaluating models in a reproducible way.
- `minerva.samplers`: Data samplers to allow for more complex sampling strategies during data loading.
- `minerva.transforms`: Data transformations and augmentations.
- `minerva.utils`: Utility functions and classes.

## Usage

For more information on how to use Minerva, check the [documentation for example notebooks](docs/notebooks).


## Testing

To run the tests, you must install the development dependencies:

```bash
pip install -e .[dev]
```

### Unit Tests

Then, you can run the tests using the following command:

```bash
pytest tests/
```

### Coverage

To generate a coverage report (in HTML), you can use the following command:

```bash
pytest --cov=minerva --cov=tests --cov-report=term --cov-report=html
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or concerns, please open an issue on our [GitHub issue tracker](https://github.com/discovery-unicamp/Minerva/issues).

## Contribute

If you want to contribute to this project make sure to read our [Code of Conduct](CODE_OF_CONDUCT.md) and [Contributing guide](CONTRIBUTING.md) pages.

## Acknowledgements

This project is maintained by Gabriel Gutierrez, Otávio Napoli, Fernando Gubitoso Marques, and Edson Borin.
