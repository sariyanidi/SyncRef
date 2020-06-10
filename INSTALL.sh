#!/bin/bash
pip3 install `cat requirements.txt`

# Compile miniball
git clone https://github.com/weddige/miniball.git
cd miniball
python setup.py install
cd ..

# Compile some cython modules
cd modules/cython_modules/
python setup.py build_ext --inplace

cd ../..

