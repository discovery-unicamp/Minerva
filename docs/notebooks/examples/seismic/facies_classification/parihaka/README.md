# Finetunning models

Each model has a finetune script in this folder, named `finetune_{model_name}.py`. All models follows the same API for training, with a few customization in some parameters (*e.g.* `img_size`) in order to fit the model architecture. The scripts are self-contained and can be run independently. For instance, to train DinoV2 (with pup head), one can use `python finetune_dinov2_pup.py`.

All scripts have the `train` function, which is responsible for training the model and saving it in to `logs` folder in a structure like `logs/{model_name}/seam_ai`, which can later be used for inference and analysis.

## Finetunning and evaluation

1. Use `ray_supervised_finetune.py` to finetune all models at once. This script uses Ray to parallelize the training process, that is, the `train` function of each model is run in parallel. Remember, to each new model, to include the model train function in the ray training loop in `ray_supervised_finetune.py`.
2. Once trained, you should include the model information for fineteuned models, in the `finetuned_models.py` file. This file is used to load the model and its configuration for inference and analysis.
3. Then, you should use the `evaluate.py` script perform inference on the test set. Remember to add the model function in the `evaluate.py` script. For multiple runs, you can add the multiple folders at `finetuned_models_path` variable in `main` function of `evaluate.py` script.
4. Once predictions are made, you can use the `evaluate.ipynb` that will perform the evaluation of the models, generating plots and metrics for comparison.

OK
