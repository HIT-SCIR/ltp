#! /usr/bin/env python
# -*- coding: utf-8 -*_

from distutils.core import setup
import setuptools

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
    packages=setuptools.find_packages(exclude=['tests', 'examples', 'docs']),
    install_requires=[
        'torch>=1.2.0',
        'torchtext==0.5.0',
        'transformers>=3.0',
        'cytoolz~=0.10.1',
        'pygtrie==2.3.3'
        'tqdm',
        'toml',
        'fire',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.6.*, <4',
)
