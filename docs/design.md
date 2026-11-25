# Minerva Design

Minerva is a flexible framework built on PyTorch and PyTorch Lightning to support machine learning tasks involving time-series, seismic data, and self-supervised learning. 
It offers tools for data management, training, evaluation, and reproducibility in ML pipelines.


## Data Handling

Minerva simplifies data management using a custom wrapper around PyTorch Lightning’s `LightningDataModule`. 
This class, called `MinervaDataModule`, provides structured access to both datasets and dataloaders.

In fact, this class:

- Integrates seamlessly with PyTorch Lightning's `Trainer`.
- Supports any map-style PyTorch dataset.
- Allows full access to both datasets and dataloaders, useful for analysis purposes.
- Configurations using YAML or other external config files as well as using Python code.

### Structure

`MinervaDataModule` organizes training, validation, and testing datasets, and provides corresponding dataloaders.
Thus, user must instantiate a `MinervaDataModule` by passing the datasets and other configurations (e.g., batch size, shuffling, etc.).

For instance, to create a `MinervaDataModule` with a torch-like map-style dataset, using train, validation, and test datasets, the user can do the following:

```python
from minerva.data.data_modules.base import MinervaDataModule

train_ds = MyTrainDataset()
val_ds = MyValDataset()
test_ds = MyTestDataset()

data_module = MinervaDataModule(
    train_dataset=train_ds,
    val_dataset=val_ds,
    test_dataset=test_ds,
    batch_size=16,
    drop_last=True
)
```

User can acess the datasets and dataloaders using the following methods and properties:

| Method/Property     | Purpose                                   |
|---------------------|-------------------------------------------|
| `train_dataset`     | Dataset used for training                 |
| `val_dataset`       | Dataset used for validation               |
| `test_dataset`      | Dataset used for testing                  |
| `train_dataloader()`| Returns training dataloader               |
| `val_dataloader()`  | Returns validation dataloader             |
| `test_dataloader()` | Returns testing dataloader                |

The dataloaders are create using configurations passed to the `MinervaDataModule` constructor.
You may check the `MinervaDataModule` class for more details on the available configurations.

### Why Not Just Use `LightningDataModule`?

While `LightningDataModule` provides access only to dataloaders, `MinervaDataModule` also exposes the underlying datasets. This makes it easier to:
- Apply dataset-level transformations or analysis
- Visualize raw samples
- Customize data preparation logic

### Minerva Datasets and Readers

Minerva's design philosophy emphasizes modularity and reusability. This is particularly important when dealing with datasets that may have different storage formats or structures.
Minerva uses `MinervaDataModule` to allow training and evaluating models, create pipelines, and conduct experiments.
As shown above, this class is instantiated passing the datasets and other configurations.
The dataset can be any map-style PyTorch dataset, which is a common format for datasets in PyTorch.

Although Minerva supports any map-style dataset, it also provides a custom dataset class called `MinervaDataset` that is designed to be minimal and modular.
This class is a wrapper around the `torch.utils.data.Dataset` class and allows users to create datasets easily by specifying keyed readers and one transform pipeline, as illustrate in the example below.

![Dataset and Readers](figures/dataset_readers_2.png)

A Reader can be viewed as a component that loads a single data unit (e.g., an image, label, or time series) from a source (e.g., disk, memory, network, table, table column, numpy array, folder, etc.) in a pre-defined order. In the example above, the `MinervaDataset` class uses two readers: one for loading images (`PNGReader`) and another for the labels (`TiffReader`). Readers are passed in an `OrderedDict`, and each key names the reader's output in the sample.
Once a `__getitem__` method is called, the `MinervaDataset` class loads the data units from the readers into one flat dict (`{"image": ..., "label": ...}`) and applies **one** transform pipeline to it. Finally, it returns the transformed dict.
Every transform takes and returns that dict (Minerva 1.0 data pipeline standard). A transform declares the keys it touches in `keys_to_transform`; every other key passes through unchanged. Random transforms draw their parameters once per call and share them across all listed keys, which keeps image and label in sync.
Thus, we can instantiate a `MinervaDataset` class by passing an `OrderedDict` of readers and a transform pipeline, as shown in the example below.

```python
from collections import OrderedDict
from minerva.data.readers import NumpyArrayReader
from minerva.transforms import RepeatTransform, _TransformsPipeline
from minerva.data.datasets import MinervaDataset

root_data_dir = Path("f3/data/")

train_data_reader = NumpyArrayReader(
    data=root_data_dir / "train" / "train_seismic.npy",
    data_shape=(1, 701, 255),
)

train_labels_reader = NumpyArrayReader(
    data=root_data_dir / "train" / "train_labels.npy",
    data_shape=(1, 701, 255),
)

train_dataset = MinervaDataset(
    readers=OrderedDict(image=train_data_reader, label=train_labels_reader),
    transforms=_TransformsPipeline(
        [RepeatTransform(["image"], axis=0, n_repetitions=3)]
    ),
)

print(train_dataset)
```

We will get the following output:

```
==================================================
           📂 MinervaDataset Information
==================================================
📌 Dataset Type: MinervaDataset
   └── Reader 'image': NumpyArrayReader(samples=401, shape=(1, 701, 255), dtype=float64)
   └── Reader 'label': NumpyArrayReader(samples=401, shape=(1, 701, 255), dtype=uint8)
   │
   └── Transforms: _TransformsPipeline(transforms=[RepeatTransform(...)])
   └── Total Readers: 2
==================================================
```

Thus, in the example above, where we have two readers, the `MinervaDataset` class loads the data units from both readers (the seismic data under `"image"` and the labels under `"label"`) and applies the `RepeatTransform` to the `"image"` key only. The resulting dataset contains dicts of the form `{"image": seismic_data, "label": labels}`. The pseudocode below illustrates how the `MinervaDataset` class works:

```
1. data = {key: reader[i] for each (key, reader)}
2. data = transforms(data)    # one pipeline for the whole sample
3. Return the dict
```

Models read the batch by key (`batch["image"]`, `batch["label"]`; contrastive techniques read `batch["view1_image"]` and `batch["view2_image"]`). Tuple-based PyTorch code interoperates through the `FromTupleDataset` (tuple dataset to dict) and `ToTupleDataset` (dict dataset to `(data, label)`) wrappers.

> **NOTE**: The `MinervaDataset` class is not mandatory, but it is a good practice to use it for maintainability and flexibility. A new task is a new set of readers and transforms, so there is rarely a reason to subclass it.


The example above creates a `MinervaDataset` from two readers, which could be two images, that may be used for training a model for image segmentation, for instance.
However, this is easilly extensible to any other type of data. Minerva provides a set of readers for loading data from different sources, such as:
- `NumpyArrayReader`: Loads data from a numpy array.
- `TabularReader`: Loads data from a tabular format with predefined columns, in row-order. 
- `CSVReader`: Loads data from a CSV file, with predefined columns, in row-order.
- `TiffReader`: Loads data from a TIFF file.
- `PNGReader`: Loads data from a PNG file.
- Among others.


### Transforms

Transforms live in `minerva.transforms` and follow the Minerva 1.0 data pipeline standard:

- Every sample is one flat `dict[str, Any]`; a transform takes a dict and returns a dict.
- The first constructor argument is `keys_to_transform`. Only those keys are changed; every other key passes through untouched.
- Random transforms (`RandomFlipTransform`, `RandomCropTransform`, ...) draw their parameters **once per call** and apply them to all listed keys, so an image and its label stay aligned. They accept an optional `seed`: with `seed=None` they use torch's global RNG (which `DataLoader` seeds per worker); with an integer they use a private generator (offset by the worker id) and are reproducible.
- Arrays use NumPy's layout, which is what the readers return: images are `(H, W)` or `(H, W, C)`, channels last. Spatial transforms (resize, crop, flip, rotate, blur, normalize, ...) work on that layout directly, so nothing needs to be transposed before transforming. Add a `TransposeTransform(['image'], (2, 0, 1))` at the **end** of a pipeline if the model expects channels first.
- Geometric transforms take a per-key interpolation `mode` (e.g. `mode=["bilinear", "nearest"]` for image and label). Without it, all keys use bilinear interpolation.
- `MultiViewTransform` runs a separate sub-pipeline per view and writes `{view}_{key}` (e.g. `view1_image`), which is how SSL techniques get independent augmentations. Do not list several view keys on one random transform to get different views. Seeded transforms in different views get distinct random streams automatically, so views built with the same `seed` still differ (and stay reproducible).
- Generic transforms all live in `minerva.transforms.transform`; technique-specific ones (`DINOTransform`, `TFCTransform`, `MoCoAugmentationTransform`, `ActivityImageTransform`) live next to their technique.
- One transform per job: `RandomCropTransform` pads when the crop is larger than the input and can pick the position from the label's class ratios (`label_key=`); `TransposeTransform` reorders any axes of arrays or tensors. `PadCropTransform` (now a deterministic centered `CropTransform`), `ClassRatioCropTransform` and `SwapAxesTransform` still work but emit a `FutureWarning` and **will be removed in the next major release**.
- Pipelines are built with `_TransformsPipeline([...])`. `ExternalTransform(keys, fn)` plugs a third-party callable (torchvision, albumentations) into a pipeline.

Signal and image augmentations use the same keyed contract. `GaussianBlurTransform` changes only the listed image keys; leave masks and categorical labels out. `VerticalPatchSwapTransform` shuffles stripes using one shared order across the listed data and mask keys, which must have matching spatial dimensions. `AmplitudeScalingTransform` applies a fixed scalar, while `RandomAmplitudeScalingTransform` samples one scalar per call. Amplitude scaling changes signal values, so list only continuous data keys and leave masks and class labels out.

```python
from minerva.transforms import (
    AmplitudeScalingTransform,
    GaussianBlurTransform,
    RandomAmplitudeScalingTransform,
    VerticalPatchSwapTransform,
    _TransformsPipeline,
)

image_pipeline = _TransformsPipeline([
    GaussianBlurTransform(["image"], kernel_size=5, prob=0.5),
    RandomAmplitudeScalingTransform(["signal"], scale_range=(0.8, 1.2)),
])
segmentation_pipeline = VerticalPatchSwapTransform(
    ["image", "mask"], patch_height=16
)
fixed_signal_transform = AmplitudeScalingTransform(
    ["signal"], scale_factor=0.9
)
```

```python
from minerva.transforms import (
    MultiViewTransform, RandomFlipTransform, RandomResizedCropTransform, _TransformsPipeline,
)

augment = _TransformsPipeline([
    RandomResizedCropTransform(["image", "label"], size=(128, 128), mode=["bilinear", "nearest"], seed=0),
    RandomFlipTransform(["image", "label"], possible_axis=1),  # axis 1 is the width for (H, W, C) and (H, W)
])
```


## Models

Minerva offers a collection of pre-defined models and architectures for supervised learning. Each model typically consists of two primary components (but is not mandatory):

* **Backbone**: Responsible for feature extraction.
* **Head**: Performs the task-specific prediction (e.g., classification, regression).

These components are usually implemented as separate classes to encourage flexibility and reusability, though this separation is not strictly required. All components can be built using either PyTorch or PyTorch Lightning modules and are fully compatible with standard PyTorch workflows.

The final model (used for training and evaluation) is generally a combination of a backbone and a head. This composite model is always implemented as a `LightningModule`, and it expects a `MinervaDataModule` instance as its data source. Thus, the Lightning `Trainer` can be used to train and evaluate the model, passing the `LightningModule` and `MinervaDataModule` instances to it.

### The `SimpleSupervisedModel` Class

To simplify the creation of supervised learning models, Minerva provides the `SimpleSupervisedModel` class, a high-level composition utility that allows users to define a model by simply passing in the backbone and head.

`SimpleSupervisedModel` is a subclass of `LightningModule` and internally follows this structure:

![SimpleSupervisedModel Structure](figures/simple_supervised_model.png)

The key components and parameters of `SimpleSupervisedModel` are:

* **`backbone`**: A PyTorch or PyTorch Lightning module that extracts features from the input. It can be randomly initialized or pre-trained.
* **`head (fc)`**: A task-specific fully connected (FC) head module. It can also be initialized from scratch or pre-trained.
* **`loss`**: A PyTorch-compatible loss function (e.g., `CrossEntropyLoss`, `MSELoss`). Used to compute the training loss.
* **`adapter`** *(optional)*: A callable that transforms the output of the backbone to a suitable format for the head. This is useful when there's a mismatch between backbone output and head input. The adapter can be a lambda, function, or module that receives the backbone output tensor and returns a tensor and returns the transformed tensor.
* **`flatten`** *(optional, boolean)*: If `True` (default), flattens the backbone (or adapter, if any) output before feeding it into the head. This is commonly needed when the output is multi-dimensional.

In addition to these, you can optionally pass optimizers, learning rate schedulers, and evaluation metrics to customize training behavior.

`SimpleSupervisedModel` supports a wide range of use cases, including transfer learning, and its modular structure makes it easy to extend or integrate with other components.

### Model Organization

Models in Minerva reside in the `minerva.models.nets` module and are further organized into domain-specific submodules:

* `minerva.models.nets.time_series`: Models for time-series data.
* `minerva.models.nets.image`: Models for image data.

Many of these models implement the `SimpleSupervisedModel` interface. While the backbone and head are often defined as separate classes in same file, this is not a strict requirement.


## Self-Supervised Learning (SSL)

Minerva also includes a suite of self-supervised learning (SSL) techniques, available in the `minerva.models.ssl` module.

In order to interoperate with Lightning Trainer and other components, all SSL methods are implemented as subclasses of `LightningModule`. Thus, they are all considered models and are compatible with the `MinervaDataModule` class, which is used to load the data for training and evaluation.
Models read the batch by key through the helpers in `minerva.models.batch` (`get_input`, `get_target`, `get_views`): supervised models use `batch["image"]` and `batch["label"]`, and contrastive techniques (SimCLR, BYOL, SimSiam, BarlowTwins, ...) use `batch["view1_image"]` and `batch["view2_image"]`, which a `MultiViewTransform` produces. Legacy tuple batches are still accepted.
Wherever possible, SSL models follow a compositional design, allowing users to pass backbone modules during initialization for greater flexibility and reusability, although this is not a strict requirement.


## Constructor-Based Initialization

Minerva uses a constructor-based approach to initialize its core components, such as models, datasets, and data modules. Wherever possible, components are configured via constructors that accept parameters, enabling flexible customization and consistent behavior.

This design makes it easy to instantiate components programmatically or from configuration files (e.g., YAML). As a result, Minerva components often expose a large number of parameters. To maintain usability, we prioritize clear documentation of all parameters and their default values.

## Training Loop

Minerva uses PyTorch Lightning's `Trainer` class to handle the training loop. Thus, any customizations or configurations for the training loop can be done using the `Trainer` class. From supervised learning to self-supervised learning, Minerva's models are designed to be compatible with the `Trainer` class, using `MinervaDataModule` for data loading.

This design allows using PyTorch Lightning's built-in features, such as:
- Lightning callbacks
- Logging
- Gradient accumulation and clipping
- Mixed precision training
- Distributed training
- Automatic accelerator and device placement



## Loading Models Partially

In PyTorch, checkpoints store a model’s `state_dict`, an **ordered dictionary** containing all of the model's parameters (typically derived from class attributes). This structure allows for selective loading of specific components, making it particularly useful for tasks such as transfer learning or fine-tuning.

For instance, when training a SimCLR SSL model with a DeepLabV3 backbone, the checkpoint usually includes the full model, both the backbone and the projection heads. However, when reusing the backbone for a different task, it’s often desirable to load only the backbone parameters and exclude the rest (e.g., the projection heads). This selective loading is crucial for transfer learning, where you want to leverage pre-trained weights without carrying over task-specific parameters.

Minerva provides a utility class called `FromPretrained` to facilitate this selective loading process. This class allows loading a model from a checkpoint while filtering out unwanted parameters (e.g., those belonging to projection heads). It uses regular expressions to filter and rename `state_dict` keys, enabling fine-grained control over what gets loaded.

Since `FromPretrained` is a class, it uses a constructor-based approach for model initialization. You simply provide the model instance and the checkpoint path, no additional wrapper functions or external logic are required. This encapsulated design promotes reusability, readability, and maintainability. It also integrates seamlessly with other Minerva components such as the `SimpleSupervisedModel` class and YAML-based configuration files.

Below is a simple example using the `FromPretrained` class, where we load a `DeepLabV3` model and selectively initialize only its `backbone` from a checkpoint file.

```python
from minerva.models.nets.image.deeplabv3 import DeepLabV3
from minerva.models.loaders import FromPretrained

checkpoint_path = "path/to/checkpoint.pth"
model = DeepLabV3()
model = FromPretrained(
    model=model,
    ckpt_path=checkpoint_path,
    filter_keys=["^backbone.*"],
    keys_to_rename=None
)
```

In this example, `FromPretrained` loads only the parameters from the checkpoint  whose keys match the regular expression `^backbone.*`, that is, parameters that starts with `backone.` in its name. This effectively excludes any parameters associated with the `fc` (fully connected) head. The optional `keys_to_rename` argument accepts a dictionary to rename or re-map or append keys during loading, useful when adapting checkpoints across slightly different model architectures, such as when the naming conventions differ or when you want to append a prefix or suffix to the keys. You may want to check the `FromPretrained` class for more details on how to use and the available parameters.

> **Note:** The use of `FromPretrained` is optional but strongly recommended for improved flexibility and maintainability. It integrates well with Minerva's `SimpleSupervisedModel` class.

> **Note:** The return value of `FromPretrained` is an instance of the same type as the input model. So, if the input is a `DeepLabV3` instance, the returned object will also be a `DeepLabV3` instance.


## Examples

To see how to use Minerva for different tasks, you can check the [tutorials page](tutorials.rst) for complete examples of how to use Minerva and its components to train and evaluate models.
