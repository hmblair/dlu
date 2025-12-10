from setuptools import setup, find_packages
import os
import re

NAME = 'dlu'


def _version() -> str:
    with open(os.path.join(os.path.dirname(__file__), NAME, '__init__.py')) as f:
        content = f.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Cannot find version information")


VERSION = _version()
LICENSE = 'CC BY-NC 4.0'
AUTHOR = 'Hamish M. Blair'
EMAIL = 'hmblair@stanford.edu'
URL = 'https://github.com/hmblair/dlu'

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'matplotlib',
    ],
    extras_require={
        'wandb': ['wandb'],
        'all': ['wandb'],
    },
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    python_requires='>=3.9',
    description='Deep Learning Utilities for PyTorch',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
