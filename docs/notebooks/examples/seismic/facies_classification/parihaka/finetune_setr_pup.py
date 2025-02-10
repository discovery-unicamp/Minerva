#!/usr/bin/env python
# coding: utf-8

# # Fine-tunning SETR PUP on Parihaka Dataset

# ## Imports

# In[1]:


from common import get_data_module, get_trainer_pipeline
from minerva.models.nets.image.setr import SETR_PUP
from torchmetrics import JaccardIndex


# ## Variaveis

# In[2]:

def train():
    root_data_dir = "/workspaces/HIAAC-KR-Dev-Container/shared_data/seam_ai_datasets/seam_ai/images"
    root_annotation_dir = "/workspaces/HIAAC-KR-Dev-Container/shared_data/seam_ai_datasets/seam_ai/annotations"
    ckpt_file = "/workspaces/HIAAC-KR-Dev-Container/shared_data/notebooks_e_pesos/SETR_pesos/vit_large_p16_new.pth"

    img_size = (1008, 784)          # Change this to the size of the images in the dataset
    model_name = "setr_pup"       # Model name (just identifier)
    dataset_name = "seam_ai"        # Dataset name (just identifier)
    single_channel = False          # If True, the model will be trained with single channel images (instead of 3 channels)

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
        single_channel=single_channel
    )

    data_module


    # In[4]:


    # Just to check if the data module is working
    data_module.setup("fit")
    train_batch_x, train_batch_y = next(iter(data_module.train_dataloader()))
    train_batch_x.shape, train_batch_y.shape


    # ## **** Create and Load model HERE ****

    # In[5]:


    model = SETR_PUP(
        image_size=img_size,
        num_classes=6,
        load_backbone_path=ckpt_file,
        original_resolution=(256, 576),
        train_metrics={"IoU": JaccardIndex(task="multiclass", num_classes=6)},
        val_metrics={"IoU": JaccardIndex(task="multiclass", num_classes=6)},
        test_metrics={"IoU": JaccardIndex(task="multiclass", num_classes=6)},
    )


    model


    # ## Pipeline

    # In[6]:


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


    # In[ ]:


    pipeline.run(data_module, task="fit")


    # In[ ]:


    print(f"Checkpoint saved at {pipeline.trainer.checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    train()