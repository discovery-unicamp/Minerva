# Pipelines: Enhancing Efficiency and Flexibility

## Introduction

Welcome to the Pipelines section! Here, we'll explore the core functionalities and best practices for creating versatile pipelines to automate tasks efficiently. Let's delve into the features and examples that demonstrate how to leverage pipelines effectively.

## 1. Reproducibility

- **Initialization and Configuration**: Pipelines are initialized using the `__init__` method, allowing configuration of common elements. All parameters passed to the class constructor are stored in the `self.hparams` dictionary, facilitating reproducibility and serialization. You can exclude specific parameters using the `ignore` parameter in the `__init__` method to enhance reproducibility.

- **ID and Working Directory**: Each pipeline instance is assigned a unique identifier (`id`) upon initialization, aiding in tracking and identification. Pipelines also have a designated working directory for organizing generated files, maintaining a clean project structure.

- **Public Interface**: Pipelines offer the `run` method as the public interface for execution. This method encapsulates the pipeline's logic and returns the output. Additionally, public attributes are implemented as read-only properties, ensuring a consistent state during execution.

## 2. Composition

- **Combining Pipelines**: Pipelines can be composed of other pipelines, enabling the creation of complex workflows from simpler components. This modularity enhances flexibility and scalability in pipeline design.

## 3. Integration with CLI

- **Seamless CLI Integration**: Pipelines integrate seamlessly with `jsonargparse`, facilitating the creation of command-line interfaces (CLI) for easy configuration and execution. Configuration can be provided via YAML files or directly through CLI run arguments, enhancing user accessibility.

## 4. Logging and Monitoring

- **Execution Log**: Pipelines maintain a log of their executions, providing a comprehensive record of activities. The `status` property offers insights into the pipeline's state, facilitating monitoring and troubleshooting.

## 5. Clonability

- **Cloning Pipelines**: Pipelines are clonable, enabling the creation of independent instances from existing ones. The `clone` method initializes a deep copy, providing a clean slate for each clone.

## 6. Parallel and Distributed Environments

- **Parallel and Distributed Execution**: Pipelines support parallel and distributed execution, enabling faster processing of tasks and efficient resource utilization. This scalability enhances performance in large-scale processing environments.

## SimpleLightningPipeline Example

In this section, we'll focus on the `SimpleLightningPipeline`, a powerful tool designed for PyTorch Lightning models. Let's explore an example of using this pipeline to train a model for computing seismic attributes:

### Configuration Setup

Start by creating a configuration file (`config.yaml`) with parameters for the model, trainer, data, and other pipeline settings.

### Running the Pipeline

Execute the pipeline using the configuration file:

```bash
python minerva/pipelines/simple_lightning_pipeline.py --config config.yaml
```

You can also use pre-configured files for training or evaluation:

```bash
# Train
python minerva/pipelines/simple_lightning_pipeline.py --config configs/pipelines/lightning_pipeline/unet_f3_reconstruct_train.yaml 

# Evaluate
python minerva/pipelines/simple_lightning_pipeline.py --config configs/pipelines/lightning_pipeline/unet_f3_reconstruct_evaluate.yaml 
```

## Configuration Files

Our modular approach to configuration files provides flexibility and organization. Configuration files are structured in directories for models, data, callbacks, loggers, trainers, and pipelines, allowing easy customization and reuse.

## Additional Notes

- Pipelines maintain logs for tracking progress and ensuring reproducibility.
- Configuration files are modular, allowing users to create custom configurations for different pipeline components.
- Extending the `SimpleLightningPipeline.evaluate` method enables customization for complex evaluation tasks beyond torchmetrics API capabilities.
- Typing annotations ensure variable clarity and facilitate seamless integration with jsonargparse for CLI configuration.

## Conclusion

Pipelines are powerful tools for automating tasks efficiently. By following best practices and leveraging versatile pipelines like `SimpleLightningPipeline`, you can streamline your workflow and achieve reproducible results with ease. Happy pipelining!

Feel free to explore more examples and documentation for detailed insights into pipeline usage and customization.
