# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../../esipy'))

project = 'ESIpy'
copyright = '2024, Joan Grèbol-Tomàs, Eduard Matito, Pedro Salvador'
author = 'Joan Grèbol-Tomàs, Eduard Matito, Pedro Salvador'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_copybutton', 'sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx.ext.githubpages', 'sphinx.ext.intersphinx', 'sphinx.ext.todo', 'sphinx.ext.viewcode',
              'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'insegel'
html_static_path = ['_static']
