# PawlikMorph-LSST
[![Documentation Status](https://readthedocs.org/projects/pawlikmorph-lsst/badge/?version=latest)](https://pawlikmorph-lsst.readthedocs.io/en/latest/?badge=latest) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/lewisfish/PawlikMorph-LSST.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lewisfish/PawlikMorph-LSST/context:python)

Translation and optimisation of SEDMORPH's [PawlikMorph](https://github.com/SEDMORPH/PawlikMorph) IDL code for analysing images of galaxies from SDSS data release 7

Replicates the ability to prepare images, generate object binary masks and calculate the A, As, As90 asymmetry parameters, and Sersic parameters.

## Usage

See the [Docs](https://pawlikmorph-lsst.readthedocs.io/en/latest/) for full information.

python imganalysis.py [-h] [-A] [-As] [-Aall] [-sersic] [-spm] [-sci] [-li]
                      [-lif] [-f FILE] [-fo FOLDER] [-src {SDSS,LSST}]
                      [-s IMGSIZE] [-cc CATALOGUE] [-ns NUMSIG]
                      [-fs {1,3,5,7,9,11,13,15}] [-par {multi,parsl,none}]
                      [-n CORES] [-m] [-cas]



 - -h, shows the help screen
 - -f FILE, File which contains list of images, and RA DECS of object in image
 - -fo FOLDER, Path to folder to save script outputs
 - -A, runs the asymmetry calculation
 - -As, Runs the shape asymmetry calculation
 - -Aall, Runs all the implmented asymmetry calculations
 - -cas, Calculate CAS parameters
 - -spm, Save calculated binary pixelmaps
 - -sci, Save cleaned image
 - -li, Use larger image cutouts to estimate sky background
 - -src, Source of the image
 - -cc, Check if any object in the provided catalogue occludes the analysed object
 - -sersic, Calculate Sersic profile
 - -fs, Size of kernel for mean filter
 - -ns, Radial extent to which mask out stars if a catalogue is provided
 - -par, Choose which library to use to parallelise script, {multi, parsl, none}
 - -n, Number of cores to use
 - -m, use precomputed masks
 
 Example
  - python imganalysis.py --file images.csv -fo sample -Aall -sersic -sci -spm -par parsl -n 8 -cas
  - This will read images from images.csv, generate a folder sample/output where pixelmaps of the object, clean images, and calculated parameters are stored.

## Installation

First clone this repo. Then either use pip or conda to install dependencies:
  - pip install --user --requirement requirements.txt
  
  Or
  - conda env create -f environment.yml

## Requirments
 - Python 3.7+
 - Numpy 1.15.4+
 - Numba 0.38.1+
 - Astropy 3.0.3+
 - Scikit-image 0.14.0+
 - Pandas 0.25.3+
 - photutils 0.7.1
 - scipy 1.3.2+
 - parsl 0.9.0+

 
 If diagnostic.py is used 
 - matplotlib
