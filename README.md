# Minerva

[![Continuous Test](https://github.com/discovery-unicamp/Minerva/actions/workflows/continuous-testing.yml/badge.svg)](https://github.com/discovery-unicamp/Minerva/actions/workflows/python-app.yml)

Welcome to Minerva, a comprehensive framework designed to enhance the experience of researchers training machine learning models. Minerva allows you to effortlessly create, train, and evaluate models using a diverse set of tools and architectures.

Featuring a robust command-line interface (CLI), Minerva streamlines the process of training and evaluating models. Additionally, it offers a versioning and configuration system for experiments, ensuring reproducibility and facilitating comparison of results within the community.

## Description

This project aims to provide a robust and flexible framework for researchers working on machine learning projects. It includes various utilities and modules for data transformation, model creation, and analysis metrics.

### Features

Minerva offers a wide range of features to help you with your machine learning projects:

- **Model Creation**: Minerva offers a variety of models and architectures to choose from, including pre-trained models and custom models.
- **Training and Evaluation**: Minerva provides tools to train and evaluate your models, including loss functions, optimizers, and evaluation metrics.
- **Data Transformation**: Minerva provides tools to preprocess and transform your data, including data loaders, data augmentation, and data normalization.
- **Command-Line Interface (CLI)**: Minerva offers a CLI to streamline the process of training and evaluating models.
- **Modular Design**: Minerva is designed to be modular and extensible, allowing you to easily add new features and functionalities.
- **Reproducibility**: Minerva ensures reproducibility by providing tools for versioning, configuration, and logging of experiments.
- **Experiment Management**: Minerva allows you to manage your experiments, including versioning, configuration, and logging.
- **SSL Support**: Minerva supports SSL (Semi-Supervised Learning) for training models with limited labeled data.

### Near Future Features

- **Hyperparameter Optimization**: Minerva will offer tools for hyperparameter optimization powered by Ray Tune.
- **PyPI Package**: Minerva will be available as a PyPI package for easy installation.

## Installation

### Install Locally

To install Minerva, you can use pip:

```sh
pip install .
```

### Get container from Docker Hub

```sh
docker pull gabrielbg0/minerva:latest
```

## Usage

You can ether use Minerva's modules directly or use the command line interface (CLI) to train and evaluate models.

### CLI

To train a model using the CLI, you can use any of the available pipelines. For example, to train a simple model using the Lightning module, you can run the following command:

```sh
python minerva/pipelines/simple_lightning_pipeline.py --config config.yaml
```

### Modules

You can also use Minerva's modules directly in your code. Just import the module you want to use and call the desired functions.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/discovery-unicamp/Minerva/blob/main/LICENSE) file for details.

## Contact

For any questions or concerns, please open an issue on our [GitHub issue tracker](https://github.com/discovery-unicamp/Minerva/issues).

## Contribute

If you want to contribute to this project make sure to read our [Code of Conduct](https://github.com/discovery-unicamp/Minerva/blob/main/CODE_OF_CONDUCT.md) and [Contributing](https://github.com/discovery-unicamp/Minerva/blob/main/CONTRIBUTING.md) pages.

## Acknowledgements

This project is maintained by Gabriel Gutierrez, Otávio Napoli, Fernando Gubitoso Marques, and Edson Borin.
