# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os  # Add os import
import sys  # Add sys import

# Make project source code findable by Sphinx - updated for new structure
sys.path.insert(0, os.path.abspath("../../src"))

project = "Chungoid"
copyright = "2025, Chungoid Team"
author = "Chungoid Team"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.napoleon",  # Support Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.githubpages",  # Creates .nojekyll file for GitHub Pages
    "sphinx_rtd_theme",  # Add the theme extension itself
    "myst_parser",  # Support for Markdown files
]

# Add support for Markdown files
source_suffix = {
    '.rst': None,
    '.md': None,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster' # Default theme
html_theme = "sphinx_rtd_theme"  # Use ReadTheDocs theme
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    'repository_url': 'https://github.com/chungoid/chungoid',
    'use_repository_button': True,
}

# -- Autodoc configuration ---------------------------------------------------
autodoc_member_order = "bysource"  # Order members by source code order
