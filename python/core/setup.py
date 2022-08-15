import os
from setuptools import setup, find_packages

project_dir, _ = os.path.split(__file__)

with open(os.path.join(project_dir, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ltp_core",
    version="0.1.0",
    author="Yunlong Feng",
    author_email="ylfeng@ir.hit.edu.cn",
    url="https://github.com/HIT-SCIR/ltp",
    description="Language Technology Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch>=1.0.0",
        "ltp_extension>=0.1.0",
        "transformers>=4.0.0",
    ],
    extras_require={
        "train": [
            # pytorch-lightning
            "pytorch-lightning>=1.0.0",
            "torchmetrics>=0.7.0",
            # datasets
            "datasets>=1.0.0",
            # hydra
            "rich",
            "pyrootutils",
            "hydra-core>=1.1.0",
            "hydra-colorlog>=1.1.0",
        ]
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
    ],
    packages=find_packages(),
    include_dirs=["ltp_core"],
    python_requires=">=3.6.*, <4",
    zip_safe=True,
)
