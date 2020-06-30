**********
ImageUtils
**********

This module contains several utility function that operate on the image.

maskstarSEG "cleans" the image of stars and other objects that emit light if they do not occlude the segmentation map.

maskstarPSF uses the star catalogue and information on the PSF in the FITS image header if both are available in order to mask out stars using the PSF.
maskSstarPSF is experimental and has not been fully developed.
The method first tries to read in information on the PSF and other viewing parameters. These values are used to `calculate <https://classic.sdss.org/dr7/algorithms/fluxcal.html#counts2mag>`_ the sky's magnitude. It then uses the objects that occlude the segmentation map as calculated by :doc:`objectMasker` module, and calculates the radius of that object based upon its psfMAG_r. The algorithm then masks that object out to its radius :math:`\times` {numsig}, where numsig is the number factor tha allows the user some control of this process.
If the adaptive option is enabled then the algorithm calculates the radius based upon following a line of flux from the centre of the object in the opposite direction of the object of interests centre, and checking for a discontinuity in the flux. It then extends the masking out until there is no discontinuity in the flux.


.. automodule:: pawlikMorphLSST.imageutils
    :members:
    :undoc-members:
    :noindex: