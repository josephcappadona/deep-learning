"""Adapted from https://github.com/pypa/sampleproject/blob/master/setup.py"""
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='projectlib',
    version='1.3.1',
    description='A sample Python project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/josephcappadona',
    author='Joseph Cappadona',
    author_email='josephcappadona27@gmail.com',
    classifiers=[ 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='deep-learning cnn autoencoders',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.5, <4',
    install_requires=[
        'tensorflow',
        'numpy',
        'sklearn',
        'matplotlib',
        'munch',
        'jupyter',
        'Pillow',
    ],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
    package_data={},
    data_files=[],
    entry_points={},
    project_urls={},
)
