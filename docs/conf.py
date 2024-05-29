# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'minerva'
copyright = '2024, Unicamp'
author = 'Discovery'

source_suffix = ['.rst']
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '**.ipynb_checkpoints',
    "**ipynb_checkpoints"
]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc.typehints',
    'sphinx.ext.mathjax',
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', '.git']

####### Auto API
autoapi_type = 'python'
autoapi_dirs = ['../minerva/']
autoapi_member_order = 'alphabetical'
autoapi_python_use_implicit_namespaces = True
autoapi_python_class_content = 'both'
autoapi_file_patterns = ['*.py']
autoapi_generate_api_docs = True
autoapi_add_toctree_entry = False
# source_suffix = '.rst'
autodoc_typehints = 'description'

######## NBSPHINX
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_codecell_lexer = 'python3'
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
htmlhelp_basename = 'minerva_docs'
htmlhelp_basename = 'minerva_docs'
source_encoding = 'utf-8'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']