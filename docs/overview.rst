********
Overview
********

For a given image, pawlikMorph-LSST will calculate a segmentation map of the galaxy of interest, subtract the sky background, then analyse the image for the following statistics:

* Gini index
* M20
* CAS
* As
* Sérsic index

There are also options to provide pawlikMorph-LSST with a star catalogue to "clean" the image of external sources, so that they do not interfere with the morphology statistics.

The overall aim of this package is to be able to analyse the images that the LSST telescope will produce.
This will involve being able to interface with the LSST data via Edinburgh's LSST servers, using LSST data Butler.

This package is based upon M. Pawlik's IDL code, and some functions are direct translations from IDL to python.