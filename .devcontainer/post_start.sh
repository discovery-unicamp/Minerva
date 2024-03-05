#!/bin/bash

echo "Installing minerva"
pip install -e .

# Add tmux options
echo -e "set -g default-terminal \"screen-256color\"\nset -g mouse on\nbind e setw synchronize-panes on\nbind E setw synchronize-panes off" >> ~/.tmux.conf

# remove full path from prompt
sed -i '/^\s*PS1.*\\w/s/\\w/\\W/g' ~/.bashrc
