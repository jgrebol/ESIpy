import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
project = 'ESIpy'
copyright = '2024, Joan Grèbol-Tomàs, Eduard Matito, Pedro Salvador'
author = 'Joan Grèbol-Tomàs, Eduard Matito, Pedro Salvador'
release = '1.0.6'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_copybutton',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'autoapi.extension',
    'sphinx.ext.autodoc.typehints',
    'sphinx_last_updated_by_git',
    'sphinxcontrib.bibtex',
    'sphinx.ext.mathjax',
    'sphinx_favicon'
]

templates_path = ["_templates"]
exclude_patterns = []
pygments_style = "sphinx"

# Updated MathJax path to a more reliable CDN
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "insegel"
html_title = "ESIpy"

# Path to your static files
html_static_path = ["_static"]

# Sphinx automatically looks inside the folders in html_static_path
# This points to _static/css/custom.css
html_css_files = ["css/custom.css"]

# Path to logo relative to conf.py
html_logo = "_static/logoesipy.png"

# Insegel theme-specific options (often helps if html_logo isn't picked up)
html_theme_options = {
    "logo": "logoesipy.png",
}

# BibTeX configuration
bibtex_bibfiles = ["_static/references.bib"]
bibtex_encoding = 'latin'

# Favicon configuration (pointing to files inside _static)
favicons = [
    "_static/favicon-16x16.png",
    "_static/favicon-32x32.png",
    "_static/favicon.ico",
]

# -- Extension configuration -------------------------------------------------
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autoapi_dirs = ["../../esipy"]
