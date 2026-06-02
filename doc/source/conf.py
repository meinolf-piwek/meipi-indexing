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

sys.path.append(str(Path("../..", "src").resolve()))

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
]
autodoc_mock_imports = [
    'cupy',
    'libxmp',
    'nvidia',
    'nvidia.dali',
    'nvidia.dali.fn',
    'nvidia.dali.fn.readers',
    'nvidia.dali.data_node',
    'nvidia.dali.plugin',
    'nvidia.dali.plugin.base_iterator',
    'nvidia.dali.plugin.pytorch',
    'pillow_heif',
    'torch',
    'torch.utils',
    'torch.utils.data',
    'torchvision',
    'transformers',
    'transformers.image_utils',
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
}

templates_path = ['_templates']
exclude_patterns = []

language = 'de'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'classic'
html_static_path = ['_static']
