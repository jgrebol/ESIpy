# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

def setup(app):
    app.add_css_file('css/custom.css')

html_context = {
    'css_files': ['_static/css/custom.css'],
}

project = 'ESIpy'
copyright = '2024, Joan Grèbol-Tomàs, Eduard Matito, Pedro Salvador'
author = 'Joan Grèbol-Tomàs, Eduard Matito, Pedro Salvador'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_copybutton', 'sphinx.ext.doctest',
              'sphinx.ext.githubpages', 'sphinx.ext.intersphinx', 'sphinx.ext.todo', 'sphinx.ext.viewcode',
              'sphinx.ext.napoleon', 'sphinx.ext.autosummary', 'sphinx.ext.autosectionlabel',
              'autoapi.extension', 'sphinx.ext.autodoc.typehints', 'sphinx_last_updated_by_git',
              'sphinxcontrib.bibtex', 'sphinx.ext.mathjax', 'sphinx.ext.autosectionlabel', 'sphinx_favicon']

templates_path = ["_templates"]
exclude_patterns = []
pygments_style = "sphinx"
bibtex_bibfiles = ["references.bib"]
mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "insegel"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "logoesipy.png"
html_title = "ESIpy"

favicons = [
    "favicon-16x16.png",
    "favicon-32x32.png",
    "favicon.ico",
]

autodoc_typehints = "description"
autodoc_class_signature = "separated"
autoapi_dirs = ["../../esipy"]
