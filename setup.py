import sys
from setuptools import setup, find_packages

setup(
    name='pytorch_wavelet',
    version='1.2',
    packages=['pytorch_wavelet'],
    author=['Michael Kellman'],
    author_email='michael.kellman@ucsf.edu',
    license='BSD',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['torch>=1.7.0']
)