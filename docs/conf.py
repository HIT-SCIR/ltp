import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'LTP4'
copyright = '2020, Feng Yunlong'
author = 'Feng Yunlong'

from ltp import __version__ as version

release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon'
]

autodoc_default_options = {
    'members': True,
    'show-inheritance': False,
    'member-order': 'bysource',
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'none'
add_module_names = False

templates_path = ['templates']
language = 'zh'
exclude_patterns = []
html_theme = 'alabaster'
html_static_path = ['static']
source_suffix = ['.rst', '.md']


def autodoc_skip_member(app, what, name, obj, skip, options):
    exclusions = (
        'register', 'weak_register', 'hook',
        'by_name', 'list_available', 'is_registered',
        'iter_registered', 'from_params', 'default_implementation', 'forward'
    )
    exclude = name in exclusions
    return skip or exclude


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)
