# Introduction
The codebase of the CVPR'20 paper, titled ``Discovering Synchronized Subsets of Sequences: A Large Scale Solution''.


![image](https://sariyanidi.com/wp-content/uploads/2023/10/syncref-1024x310.jpg)

If you use this code in your research papers, please cite the work as follows.

``` @InProceedings{Sariyanidi_2020_CVPR,
author = {Sariyanidi, Evangelos and Zampella, Casey J. and Bartley, Keith G. and Herrington, John D. and Satterthwaite, Theodore D. and Schultz, Robert T. and Tunc, Birkan},
title = {Discovering Synchronized Subsets of Sequences: A Large Scale Solution},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Installation

The ```SyncRef``` software runs on python (python 3) and is installed as follows. (We assume that pip3 is installed on your system.) Optionally, you can install on a virtual environment by running the following two commands prior to installation:

```
virtualenv -p python3 syncrefenv
source syncrefenv/bin/activate
```

To install SyncRef, simply clone this repository and run

```
chmod +x INSTALL.sh
./INSTALL.sh
```

## Running the code
To run a demo of the Syncref software, you can simply excute the command
```
python demo.py
```

If you installed on a virtual environment, make sure that you activated the virtual environment prior to running the demo by executing the following command

```
source syncrefenv/bin/activate
```

If you successfully run the software, you should see a figure depicting the identified synchronized sequences and the run time of the algorithm printed on the command line.

## Versions of dependent packages 
We have tested with the following verions
```
cython 0.29.16
numpy 1.18.2
pandas 1.0.3
sklearn 0.22.2.post1
matplotlib 3.2.1
scipy 1.4.1
```
