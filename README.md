# PawlikMorph-LSST
Translation and optimisation of SEDMORPH's [PawlikMorph](https://github.com/SEDMORPH/PawlikMorph) IDL code for analysing images of galaxies from SDSS data release 7

Currently only replicates the ability to generate aperture pixel maps 

## Usage

./imganalysis.py [-h] [-f FILE] [-fo FOLDER] [-aperpixmap]

 - -h, shows the help screen
 - -f FILE, Path to a single image for analysis
 - -fo FOLDER, Path to folder of images for analysis
 - -aperpixmap, Generate aperture pixel maps

## Requirments
 - Python 3.6.5+
 - Numpy 1.15.4+
 - Numba 0.38.1+
 - Astropy 3.0.3+
 
 ## TODO
  - Calculate Asymmetry
  - Calculate outer asymmetery
  - Calculate shape asymmetry
  - Analysis LSST images and other similar images
  - Run on LSST's data centre in Edinburgh
  - Make it all fast
  - Notebook integration
  - Docker integration?
