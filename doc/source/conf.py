# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from pathlib import Path
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'meipi-indexing'
copyright = '2026, Meinolf Piwek'
author = 'Meinolf Piwek'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.append(str(Path('../..', 'src/meipi').resolve()))

extensions = ['sphinx.ext.autosummary', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']
autodoc_mock_imports = ['nvidia.dali']

templates_path = ['_templates']
exclude_patterns = []

language = 'de'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'classic'
html_static_path = ['_static']
