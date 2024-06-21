.. _installation:

==========================
Installation Guide
==========================

Minerva is a standard Python package that can be installed using pip. We also provide a Docker image for easy deployment.

Pip Installation
---------------------

To install the latest stable version of Minerva, you can use pip to install it 
from github main branch using the following command:

.. code-block:: bash
    
    pip install git+https://github.com/discovery-unicamp/Minerva.git


Using Docker
---------------------

We provide a Docker image for easy deployment. You can pull the image from 
Docker Hub using the following command:

.. code-block:: bash

    docker pull gabrielbg0/minerva:latest


Development Install
---------------------

If you want to install Minerva for development purposes, you can clone the repository and install it using pip:

.. code-block:: bash

    git clone https://github.com/discovery-unicamp/Minerva.git Minerva
    cd Minerva
    pip install -e ".[dev]"


If you want to install minerva on your local machine, outside any container, we 
recommend using a virtual environment to install Minerva. You can create a virtual environment using the following command:

.. code-block:: bash

    cd Minerva
    python3 -m venv venv
    source venv/bin/activate
    pip install -e ".[dev]"

If you want to build the documentation, you can install the documentation dependencies using the following command:

.. code-block:: bash

    pip install -e ".[dev,docs]"


Running the tests
~~~~~~~~~~~~~~~~~~

After installing Minerva, you can run the tests using standard Python testing tools, such as `pytest`:

.. code-block:: bash

    pytest
