#!/usr/bin/env python
# coding: utf-8

# # Fine-tunning SFM Base Patch16 on Parihaka Dataset

# ## Imports

# In[1]:


from common import get_data_module, get_trainer_pipeline
from minerva.models.nets.image.vit import SFM_BasePatch16_Downstream
from minerva.models.loaders import FromPretrained


# ## Variaveis

# In[2]:

def train():
    root_data_dir = "/workspaces/HIAAC-KR-Dev-Container/shared_data/seam_ai_datasets/seam_ai/images"
    root_annotation_dir = "/workspaces/HIAAC-KR-Dev-Container/shared_data/seam_ai_datasets/seam_ai/annotations"
    ckpt_path="/workspaces/HIAAC-KR-Dev-Container/shared_data/seismic_foundation_model/pretrained_models/SFM-Base-512.pth",

    img_size = (512, 512)          # Change this to the size of the images in the dataset
    model_name = "sfm-base-patch16"       # Model name (just identifier)
    dataset_name = "seam_ai"        # Dataset name (just identifier)
    single_channel = True          # If True, the model will be trained with single channel images (instead of 3 channels)

    log_dir = "/workspaces/HIAAC-KR-Dev-Container/Minerva-Dev/docs/notebooks/examples/seismic/facies_classification/parihaka/logs"              # Directory to save logs
    batch_size = 1                  # Batch size    
    seed = 42                       # Seed for reproducibility
    num_epochs = 100                # Number of epochs to train
    is_debug = False                 # If True, only 3 batch will be processed for 3 epochs
    accelerator = "gpu"             # CPU or GPU
    devices = 1                     # Num GPUs


    # ## Data Module

    # In[3]:


    data_module = get_data_module(
        root_data_dir=root_data_dir,
        root_annotation_dir=root_annotation_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        single_channel=single_channel, 
    )

    data_module


    # In[4]:


    # Just to check if the data module is working
    data_module.setup("fit")
    train_batch_x, train_batch_y = next(iter(data_module.train_dataloader()))
    train_batch_x.shape, train_batch_y.shape


    # ## **** Create and Load model HERE ****

    # In[5]:


    model = SFM_BasePatch16_Downstream(
        img_size=(512, 512),
        num_classes=6,
        in_chans=1
    )

    model


    # In[6]:


    model = FromPretrained(
        model=model,
        ckpt_path="/workspaces/HIAAC-KR-Dev-Container/shared_data/seismic_foundation_model/pretrained_models/SFM-Base-512.pth",
        ckpt_key="model",
        strict=False,
        filter_keys=["^blocks*", "^cls_token", "^pos_embed", "^patch_embed", "^norm"],
        keys_to_rename={"": "backbone."},
        ckpt_load_weights_only=False,
        error_on_missing_keys=False
    )


    # ## Pipeline

    # In[7]:


    pipeline = get_trainer_pipeline(
        model=model,
        model_name=model_name,
        dataset_name=dataset_name,
        log_dir=log_dir,
        num_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        is_debug=is_debug,
        seed=seed,
    )


    # In[8]:


    pipeline.run(data_module, task="fit")


    # In[9]:


    print(f"Checkpoint saved at {pipeline.trainer.checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    train()