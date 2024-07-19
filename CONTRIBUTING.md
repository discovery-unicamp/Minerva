# Contributing to Minerva

## Introduction

First off, thank you for considering contributing to the project. It's people like you that make Minerva such a great tool.

Following the guidelines ensures that your contribution will be considered by the maintainers. It also helps you know what to expect from the project maintainers in return. The guidelines exist to help all of us work together in a more efficient way, creating a common language that we can all understand.

Minerva is a tool that is constantly evolving, and we are always looking for ways to improve it. We are looking for contributions in bug fixes, new features, tests and documentation improvements. If you have any questions, please don't hesitate to ask.

## Ground Rules

First of all, make sure to follow our [Code of Conduct](https://github.com/discovery-unicamp/Minerva/blob/main/CODE_OF_CONDUCT.md) and be respectful to all contributors and community members.

### Issues

* **Bug reports** should be as detailed as possible. Include the steps to reproduce the bug, the expected behavior and the actual behavior. If possible, include screenshots. There is a template available for bug reports. You should follow it as closely as possible.
* **Feature requests** should be more concise. Include the problem you are facing, the solution you are thinking about and any other information that you think is relevant. There is also a template available for feature requests you should follow as close as possible.

Any other discussion should be done in the [Discussions](https://github.com/discovery-unicamp/Minerva/discussions) tab.

### Pull Requests

* **Code changes** should be tested and documented. If you are adding a new feature, make sure to include tests for it. If you are fixing a bug, make sure to include a test that reproduces the bug when possible.
* **Commit messages** should be clear and concise. If you are fixing a bug, include the issue number in the commit message. If you are adding a new feature, include a brief description of it.

## Getting started

For changes or fixes:

1. Create a new fork for your changes
2. Make the changes needed in this fork
3. If you are ready to merge your code to the main make sure that:
   1. You wrote tests for your changes
   2. All tests pass, new and old
   3. Your code is well documented
4. Make your PR

As a rule of thumb, changes are obvious fixes if they do not introduce any new functionality or creative thinking. As long as the change does not affect functionality, some likely examples include the following:

* Spelling / grammar fixes
* Typo correction, white space and formatting changes
* Comment clean up
* Bug fixes that change default return values or error codes stored in constants
* Adding logging messages or debugging output
* Changes to ‘metadata’ files like Gemfile, .gitignore, build scripts, etc.
* Moving source files from one directory or package to another

## Making Code Contributions

Every code contribution should be made through a pull request. This applies to all changes, including bug fixes and new features. This allows the maintainers to review the code and discuss it with you before merging it. It also allows the community to discuss the changes and learn from them.

You code should follow the following guidelines:

* **Documentation**: Make sure to document your code. This includes docstrings for functions and classes, as well as comments in the code when necessary. For the documentation, we use the numpydoc style. Also make sure to update the `README` file or other metadata files if necessary.
* **Tests**: Make sure to write tests for your code. We use `pytest` for testing. You can run the tests with `python -m pytest` in the root directory of the project.
* **Commit messages**: Make sure to write clear and concise commit messages. Include the issue number if you are fixing a bug.
* **Dependencies**: Make sure to include any new dependencies in the `requirements.txt` and `pyproject.toml` file. If you are adding a new dependency, make sure to include a brief description of why it is needed.
* **Code formatting**: Make sure to run a code formatter on your code before submitting the PR. We use `black` for this.

You should also try to avoid rewriting functionality, or adding dependencies for functionalities that are already present on one of our dependencies. This would make the codebase more bloated and harder to maintain.

If you are contributing code that you did not write, you must ensure that the code is licensed under an [MIT License](https://opensource.org/licenses/MIT). If the code is not licensed under an MIT License, you must get permission from the original author to license the code under the MIT License. Also make sure to credit the original author in a comment in the code.

### Module Specific Guidelines

#### `models` module

Our models are based on the `lightning.LightningModule` class. This class is a PyTorch Lightning module that simplifies the training process. You should follow the PyTorch Lightning guidelines for writing your models. You can find more information [here](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html).

As a rule of thumb, all front facing model classes should inherit from the `LightningModule` class. Subclasses of this class can be only `torch.nn.Module` classes.

In the same way, all front facing model classes should have default parameters for the `__init__` method. This classes also should be able to receive a `config` parameter that will be used to configure the model. The config parameter should be a dictionary with the parameters needed to configure the model.

The `models` module is divided into `nets` and `ssl`:

* The `nets` module contains model architectures that can be trained in a supervised way.
* The `ssl` module contains logic and implementations for self-supervised learning techniques.

In a general way, you should be able to use a `nets` model in to a `ssl` implementation to train a model in a self-supervised way.

We strongly recommend that, when possible, you divide your model into a backbone and a head. This division allows for more flexibility when using the model in different tasks and with different ssl techniques.

## How to report a bug

### Security Vulnerabilities

If you find a security vulnerability, **do NOT** open an issue. Contact one of the core maintainers privately or email <edson@ic.unicamp.br> instead.

In order to determine whether you are dealing with a security issue, ask yourself these two questions:

* Can I access something that's not mine, or something I shouldn't have access to?
* Can I disable something for other people?

If the answer to either of those two questions are "yes", then you're probably dealing with a security issue. Note that even if you answer "no" to both questions, you may still be dealing with a security issue, so if you're unsure, just contact us.

### Other Bug Reports

You can even include a template so people can just copy-paste (again, less work for you).

When filing an issue, make sure to answer these five questions:

1. What version of Python are you using?
2. What operating system and processor architecture are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

Other than these questions, if you think any other information may be useful feel free to include it.

General questions should go to the [Discussions](https://github.com/discovery-unicamp/Minerva/discussions) tab instead of the issue tracker. The community there will answer or ask you to file an issue if you've tripped over a bug.

## How to suggest a feature or enhancement

One of our philosophies is to implement the least redundant features possible, without making our dependencies bloated. So, if there is a function within the scope of one of our dependencies that already solves what you are trying to implement you should prioritize it.

If you wish for a feature that isn't currently available in Minerva, you're likely not the only one. Many users share similar needs, especially since Minerva is still in its early stages. Please open an issue on our GitHub issues list, detailing the feature you'd like to see, why you need it, and how it should work.

## Code review process

As the core team is composed mostly of postgraduate students, we may take some time to review your PR. We will do our best to review it as soon as possible, but we may take a few days to do so. Features deemed as core to the project or critical bug fixes will be reviewed first.

## Community

If you have any questions, ideas or other topic to discuss, feel free to use the [Discussions](https://github.com/discovery-unicamp/Minerva/discussions) tab. We will be happy to help you.

We would like to thank you for your interest in contributing to Minerva. We are looking forward to your contributions.

The Minerva Team.
