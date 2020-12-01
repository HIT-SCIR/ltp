#! /usr/bin/env python
# -*- coding: utf-8 -*_

import setuptools
from distutils.core import setup

from ltp import __version__ as version

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='ltp',
    version=version,
    author='Yunlong Feng',
    author_email='ylfeng@ir.hit.edu.cn',
    url='https://github.com/HIT-SCIR/ltp',
    description='Language Technology Platform',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=[
        'docs'
        'tools',
        'tests',
        'examples',
        'config',
    ]),
    install_requires=[
        "torch>=1.2.0",
        "transformers>=3.2.0, <4",
        "pygtrie>=2.3.0, <2.5"
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.6.*, <4',
)
