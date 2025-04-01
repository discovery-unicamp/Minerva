# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = "minerva"
copyright = "2025, Unicamp"
author = "Discovery"

# source_suffix = ['.rst', '.md']
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints", "**ipynb_checkpoints"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",                      # MyST markdown parser
    "sphinx.ext.autodoc",               # Support for automatic documentation
    "autoapi.extension",                # Auto-generate API documentation
    "sphinx_rtd_theme",                 # ReadTheDocs theme
    "sphinx.ext.viewcode",              # Add "view source code" links
    "sphinx.ext.autodoc.typehints",     # Use type hints in autodoc
    "sphinx.ext.mathjax",               # Render math equations
    "nbsphinx",                         # Support for Jupyter notebooks
    "IPython.sphinxext.ipython_console_highlighting",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "Thumbs.db", ".DS_Store", ".git"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "minerva_docs"
htmlhelp_basename = "minerva_docs"
source_encoding = "utf-8"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for AutoAPI -----------------------------------------------------
autoapi_type = "python"                 # The type of the API documentation
autoapi_dirs = ["../minerva/"]          # The directories to process
autoapi_member_order = "alphabetical"   # The order of the members in the documentation
autoapi_python_use_implicit_namespaces = True   
autoapi_python_class_content = "both"
autoapi_file_patterns = ["*.py"]        # The file patterns to include
autoapi_generate_api_docs = True        # Generate the API documentation automatically
autoapi_add_toctree_entry = False       # Add the API documentation to the table of contents
autodoc_typehints = "description"       # Use type hints in autodoc
# source_suffix = '.rst'

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_execute = "never"              # Execute Jupyter notebooks during the Sphinx build. If not, the notebook and outputs should be executed before building the documentation
nbsphinx_allow_errors = True            # Allow errors in the notebooks
nbsphinx_codecell_lexer = "python3"     # The lexer to use for code cells
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]


