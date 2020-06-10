#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:41:14 2020

@author: v
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(name="cython_functions", ext_modules=cythonize('cython_functions.pyx'),)
