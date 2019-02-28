#!/usr/bin/env python

from setuptools import setup, find_packages

version = '0.1.0'

required = open('requirements.txt').read().split('\n')

setup(
    name='fleappy',
    version=version,
    description='A modular framework for the analysis of experiment data neuroscience experiments.',
    author='Jeremy T. Chang',
    author_email='jeremy.chang@mpfi.org',
    url='https://github.com/jtchang/MPFI_IC',
    packages=['fleappy.notifications', 'fleappy.tiffread', 'fleappy.imgregistration', 'fleappy.experiment', 'fleappy.roimanager', 'fleappy.metadata', 'fleappy.imageviewer', 'fleappy.analysiis'],
    install_requires=required,
    long_description='See ' + 'https://mpif-ic.readthedocs.io',
    license='MIT',
    python_requires='~=3.7'
)
