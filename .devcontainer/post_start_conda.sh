#!/bin/bash

# Install minerva in editable mode
echo "Creating conda environment..."
conda env create -f environment.yaml

if [ $? -ne 0 ]; then
    echo "Failed to create conda environment!"
    exit 1
fi

echo "conda activate minerva-dev" >> ~/.bashrc
echo "Conda environment created successfully!"

