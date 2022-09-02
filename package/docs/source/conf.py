# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


# import module
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("./package"))
# sys.path.insert(0, os.path.join(os.path.dirname((os.path.abspath('.')), 'package')))
                
print(sys.path)

# project information ( you can change this )

project = 'GrainLearning'
copyright = '2022, Hongyang Cheng, Retief Lubbe'
author = 'Hongyang Cheng, Retief Lubbe'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.coverage","sphinx_autodoc_typehints"]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

pygments_style = 'sphinx'
highlight_language = 'python3'
autodoc_member_order= 'groupwise'

add_module_names = False
autodoc_typehints="description"

html_static_path = []