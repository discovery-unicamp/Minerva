#!/bin/bash

# Add /home/vscode/.local/bin directory to PATH
export PATH="/home/vscode/.local/bin':$PATH" >> ~/.bashrc

# Install minerva in editable mode
echo "Installing minerva develop (in editable mode)..."
pip install -e .[dev]
echo "Minerva installed in editable mode"
