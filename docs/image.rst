*****
Image
*****

This module contains the Image abstract class, with the SDSS and SDSS-LSST class that inherit from it.
The Image abstract class is meant to be a base class for future expansion of the code for other sources of images.
The SDSS image class should function as a catch all for any source of FITS images, not just SDSS images. All that is needed to use the SDSS image class is a FITS image and a RA, and DEC value to pinpoint the object of interest.

The LSST image class for now only works with SDSS images in a certain folder structure. However, this is provided in order to help any future developers 

.. automodule:: pawlikMorphLSST.Image
    :members:
    :undoc-members:
    :noindex: