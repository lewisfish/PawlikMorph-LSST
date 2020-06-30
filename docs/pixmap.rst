******
Pixmap
******

This module contains the routines to calculate the segmentation map based upon the 8 connected pixel method, and methods related to the segmentation map.

To calculate a segmentation map (pixelmap), we first apply a mean filter of size {filterSize}. This filter size can be set using the --filterSize command line option when using imganalysis.py. Currently the filter size is constrained to be an odd integer between 3 and 15.

The next step is to reduce the image size by a factor of filterSize. We then check if the central pixel is brighter than the sky value, if it is not we raise an ImageError.

We then set the central pixel as part of the segmentation map and start the 8 connected algorithm. This algorithm checks the surrounding 8 pixels that are connected to the current pixel, starting with the central pixel. Each of the connecting pixels are only added to the segmentation map if and only if that pixel's flux is greater  than :math:`sky+\sigma_{sky}`. This whole process repeats for each pixel that is added to segmentation map until no more pixels can be added.

If a starmask is provided, the above process still takes place, except some of the pixels where there are stars are pre-masked out before the 8 connected algorithm starts.

The other functions in this module are methods that operate on the mask.
For instance, calcRmax, calculates the maximum distance from a pixel on the edge of the segmentation map to the central pixel.
calcMaskedFraction is only used if a star catalogue is provided. This calculates the amount of pixels masked out of object of interest due to masking of nearby stars. This is calculated by first rotating the masked output and then taking the residual of this in order to offset any artificially induced asymmetry.
Finally, checkPixelmapEdges gives a warning if any pixels in the segmentation map are on the edge of the image.

Below are the function signatures for pixelmap, and the other functions in this module.

.. automodule:: pawlikMorphLSST.pixmap
    :members:
    :undoc-members:
    :noindex: