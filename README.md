# PawlikMorph-LSST
Translation and optimisation of SEDMORPH's [PawlikMorph](https://github.com/SEDMORPH/PawlikMorph) IDL code for analysing images of galaxies from SDSS data release 7

Replicates the ability to prepare images, generate object binary masks and calculate the A, As, and As90 asymmetry parameters.

## Usage

./imganalysis.py [-h] [-f FILE] [-fo FOLDER] [-A] [-As] [-Aall] [-spm] [-sci] [-li] [-src {sdss,hsc}] [-cc CATALOGUE] [-sersic]


 - -h, shows the help screen
 - -f FILE, Path to a single image for analysis
 - -fo FOLDER, Path to folder of images for analysis
 - -A, runs the asymmetry calculation
 - -As, Runs the shape asymmetry calculation
 - -Aall, Runs all the implmented asymmetry calculations
 - -spm, Save calculated binary pixelmaps
 - -sci, Save cleaned image
 - -li, Use larger image cutouts to estimate sky background
 - -src, Source of the image
 - -cc, Check if any object in the provided catalogue occludes the analysed object
 - -sersic, Calculate Sersic profile
 
 Example
  - ./imganalysis.py -fo sample/data -Aall -spm -sci -src sdss -li -sersic
  - This will generate a folder sample/output where pixelmaps of the object, clean images, and calculated parameters are stored.

## Installation

First clone this repo. Then either use pip or conda to install dependencies:
  - pip install --user --requirement requirements.txt
  
  Or
  - conda env create -f enviroment.yml

## Requirments
 - Python 3.7+
 - Numpy 1.15.4+
 - Numba 0.38.1+
 - Astropy 3.0.3+
 - Scikit-image 0.14.0+
 - Pandas 0.25.3+
 - photutils 0.7.1+
 
 ## TODO
  - [x] Calculate Asymmetry
  - [x] Calculate shape asymmetry
  - [x] Analyse HSC data
  - [ ] Analyse LSST images
  - [ ] Run on LSST's data centre in Edinburgh
  - [ ] Make it all fast
  - [ ] Notebook integration
  - [ ] Docker integration?
