# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Add the parent directory (NeuroAI-Toolkit) to the Python path
# This allows Sphinx to find your 'neusen_reveal' package
sys.path.insert(0, os.path.abspath('../../')) # Adjust path if your docs are nested differently

# Import your package to get the version
try:
    from neursen_reveal import __version__
except ImportError:
    # Fallback if package isn't installed in the env Sphinx is running in
    print("Warning: Could not import 'neusen_reveal' to get version. Using '0.0.0'.")
    __version__ = '0.0.0' # Or some default/placeholder


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'neusen_reveal'
copyright = '2025, Nada Omran, Nouran Khattab, Youssef Shawky'
author = 'Nada Omran, Nouran Khattab, Youssef Shawky'
release = __version__ # The full version, including alpha/beta/rc tags.
version = '.'.join(__version__.split('.')[:2]) # The short X.Y version.

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# extensions coming with Sphinx (named 'sphinx.ext.*') or custom ones.
extensions = [
    'sphinx.ext.autodoc',       # To automatically document Python modules
    'sphinx.ext.napoleon',      # To parse NumPy and Google style docstrings
    'sphinx.ext.viewcode',      # To link to highlighted source code
    'sphinx.ext.githubpages',   # For easy hosting on GitHub Pages
    'sphinx.ext.autosummary',   # To generate summary tables for modules/functions
    'sphinx.ext.todo',          # For todo notes (useful during development)
    'sphinx.ext.mathjax',       # For LaTeX math
    'sphinx_gallery.gen_gallery', # If you convert notebooks to rst, otherwise not needed
    'nbsphinx', # To parse Jupyter notebooks directly
]

# nbsphinx configuration (important for example notebook)
nbsphinx_allow_errors = True 
nbsphinx_execute_arguments = [
    "--Reader.parse_language_cell_magics=True",
]

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = False # Set to True if you use NumPy style docstrings
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False # Set to True if you want types to be processed by Sphinx

sphinx_gallery_conf = {
    'examples_dirs': '../../examples',     # ← path to your tutorials (relative to conf.py)
    'gallery_dirs': 'auto_examples',       # ← output folder in build/html/
}
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo' # A modern, clean Sphinx theme (pip install furo)
# html_theme = 'sphinx_rtd_theme' # Another popular theme (pip install sphinx_rtd_theme)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_theme_options = {
    # ... any other Furo options you have ...
    "extra_navbar_links": [
        {
            "title": "Third-Party Licenses",
            "url": "third_party_licenses.html",
        },
    ]
}
# You might want to add a logo
# html_logo = '_static/your_logo.png' # Place your logo image in docs/source/_static/
