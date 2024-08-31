# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

project = 'Sentinel Segmentation'
copyright = '2024, Yibing Chen'
author = 'Yibing Chen'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',  # Adding the theme here
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Use the Read the Docs theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Additional theme options can be added here
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
