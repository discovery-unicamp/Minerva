.. _tutorials:

==========================
Examples and Tutorials
==========================

This section provides a set of tutorials to help you get started with Minerva and learn how to train models for a variety of machine learning tasks.
All tutorials are written as interactive Jupyter notebooks and are organized by topic for ease of navigation

Getting Started Notebooks
---------------------------

These introductory tutorials demonstrate how to use Minerva to train models on two representative tasks using supervised learning and state-of-the-art architectures from the literature:

- **Seismic Facies Classification**: Explore how to segment seismic images into different facies classes. This is formulated as a semantic segmentation task on 2D image data.

- **Human Activity Recognition (HAR)**: Learn how to classify human activities using time-series data from smartphone sensors (e.g., accelerometers, gyroscopes). This is modeled as a time-series classification problem.

- **Experiment API**: Understand how to use the experiment API to configure and run experiments, log results, and analyze performance metrics. This is a general-purpose tutorial (constructed over seismic facies classification example) that can be applied to any task.


..  toctree::
    :maxdepth: 1

    notebooks/seismic_facies_getting_started.ipynb
    notebooks/har_getting_started.ipynb
    notebooks/experiment_api_example.ipynb
    
