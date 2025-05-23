# Getting Started

Minerva is a framework that provides tools and libraries for developing machine learning models, particularly for seismic and time-series processing, as well as self-supervised learning tasks. It also includes utilities for conducting reproducible experiments.

Built on top of PyTorch and PyTorch Lightning, Minerva is designed to be modular and extensible, allowing users to integrate and extend its functionalities with ease.

Once installed, Minerva functions like any other Python package. You can import its modules and classes as follows:

```python
import minerva
```

## Examples and Tutorials Notebooks

Check out some getting started notebooks that provide a step-by-step guide on how to use Minerva for different tasks at [tutorials page](tutorials.rst)

## Core Modules

Minerva consists of several specialized modules, each supporting different aspects of model development:

| Module                | Description |
|-----------------------|-------------|
| `minerva.analysis`    | Tools for analyzing and visualizing model performance. |
| `minerva.callback`    | Callbacks for monitoring and logging training progress (built on PyTorch Lightning). |
| `minerva.data`        | Readers, datasets, and data modules. |
| `minerva.engines`     | Training and evaluation engines for pipelines and models. |
| `minerva.losses`      | Collection of loss functions. |
| `minerva.models`      | Predefined models and architectures for supervised and self-supervised learning. For more details, check the [minerva.models README.md](minerva/models/README.md). |
| `minerva.optimizers`  | Custom optimizers and learning rate schedulers. |
| `minerva.pipelines`   | Utilities for training and evaluating models in a reproducible manner. |
| `minerva.samplers`    | Custom data samplers to enable advanced sampling strategies during data loading. |
| `minerva.transforms`  | Data transformations and augmentations. |
| `minerva.utils`       | General utility functions and helper classes. |


### Minerva Design

Minerva is designed to be modular and extensible, allowing users to integrate and extend its functionalities with ease. Checkout the [Minerva design](design.md) to understand how Minerva is structured and how to extend it.

Also, minerva provides a powerful experimental framework that allows you to run experiments in a reproducible manner. The experimental framework is built on top of PyTorch Lightning and provides a set of tools to manage experiments, organize and log results, and evaluate model. You may refer to the [Minerva experimental framework](experiments.md) to understand how to use it.

