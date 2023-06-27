import os

from setuptools import find_packages, setup

project_dir, _ = os.path.split(__file__)

with open(os.path.join(project_dir, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ltp",
    version="4.2.14",
    author="Yunlong Feng",
    author_email="ylfeng@ir.hit.edu.cn",
    url="https://github.com/HIT-SCIR/ltp",
    description="Language Technology Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "ltp_core>=0.1.3",
        "ltp_extension>=0.1.9",
        "huggingface_hub>=0.8.0",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries",
    ],
    packages=find_packages(),
    python_requires=">=3.6, <4",
    zip_safe=True,
)
