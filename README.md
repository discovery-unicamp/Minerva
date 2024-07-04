# Minerva

[![Continuous Test](https://github.com/discovery-unicamp/Minerva-Dev/actions/workflows/continuous-testing.yml/badge.svg)](https://github.com/discovery-unicamp/Minerva/actions/workflows/python-app.yml)

Minerva is a framework for training machine learning models for researchers.

## Description

This project aims to provide a robust and flexible framework for researchers working on machine learning projects. It includes various utilities and modules for data transformation, model creation, and analysis metrics.

## Installation

### Intall Locally
To install Minerva, you can use pip:

```sh
pip install .
```
### Get container from Docker Hub

```
docker pull gabrielbg0/minerva:latest
```

## Usage

You can eather use Minerva's modules directly or use the command line interface (CLI) to train and evaluate models.

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
