#!/bin/bash

# Set Ray address
RAY_ADDRESS='192.168.1.213:6379'
WORKING_DIR='./'

# Run the finetuning script
python ray_supervised_finetune.py --ray-address=${RAY_ADDRESS} 