# PawlikMorph-LSST
Translation and optimisation of SEDMORPH's [PawlikMorph](https://github.com/SEDMORPH/PawlikMorph) IDL code for analysing images of galaxies from SDSS data release 7

Currently only replicates the ability to generate aperture pixel maps and object binary mask.

## Usage

./imganalysis.py [-h] [-f FILE] [-fo FOLDER] [-A] [-As] [-Aall] [-aperpixmap] [-spm] [-nic]

 - -h, shows the help screen
 - -f FILE, Path to a single image for analysis
 - -fo FOLDER, Path to folder of images for analysis
 - -A, runs the asymmetry calculation
 - -As, Runs the shape asymmetry calculation
 - -Aall, Runs all the implmented asymmetry calculations
 - -aperpixmap, Generate aperture pixel maps
 - -spm, Save calculated binary pixelmaps
 - -nic, Save cleaned image
 
 Example
  - ./imganalysis.py -f sample/sdsscutout_211.51-0.31_rband.fits -aperpixmap

## Installation

First clone this repo. Then either use pip or conda to install dependencies:
  - pip install --user --requirement requirements.txt
  
  Or
  - conda env create -f enviroment.yml

## Requirments
 - Python 3.6.5+
 - Numpy 1.15.4+
 - Numba 0.38.1+
 - Astropy 3.0.3+
 - Scikit-image 0.14.0+
 - [Gaussfitter](https://github.com/keflavich/gaussfitter) modified version included as gaussfitter.py
 
 ## TODO
  - [x] Calculate Asymmetry
  - [x] Calculate shape asymmetry
  - [ ] Calculate outer asymmetery
  - [ ] Analysis LSST images and other similar images
  - [ ] Run on LSST's data centre in Edinburgh
  - [ ] Make it all fast
  - [ ] Notebook integration
  - [ ] Docker integration?
