************
Installation
************

Requirements
============

PawlikMorph-LSST depends on several other packages for some features:

* Python 3.7+
* Numpy 1.15.4+
* Numba 0.38.1+
* Astropy 3.0.3+
* Scikit-image 0.14.0+
* Pandas 0.25.3+
* photutils 0.7.1+
* scipy 1.3.2+
* parsl 0.9.0+

If diagnostic.py is used:

* matplotlib

Installing the latest version
=============================

First clone the GitHub repository:

* Via commandline::

    git clone https://github.com/lewisfish/PawlikMorph-LSST.git

* Or via webpage:

    * https://github.com/lewisfish/PawlikMorph-LSST/archive/master.zip

Then install the dependancies:

* Via conda::

    conda env create -f environment.yml

* Or via pip::

    pip install --user --requirement requirements.txt