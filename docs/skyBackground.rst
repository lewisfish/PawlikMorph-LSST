*************
SkyBackground
*************

This module contains the routines used to calculate the sky background value.
It does this by first fitting a Gaussian to the object of interest in the image.

This routine carries out the following procedures:

First it checks if there are any sources of light in the image. If there are none then the algorithm raises a SkyError.

The next step is to fit a 2D Gaussian to the light source. There are two 2D Gaussian fitting routines used. The one used first is less robust but faster, and usually gives an accurate fit. However, if this fitter fails then the 2nd Gaussian fitter is used, which is robust, but a lot slower.

Next the algorithm checks the size of the sky area around the object of interest. If this area is less than 300 pixels or the radius of the Gaussian is larger than 3 times the image size, then more robust Gaussian fitter is used. This is done as most of the time the less robust fitter is used and it cant cope with objects that are on the same size scale as the image itself. 

Again the sky area is checked, this time if it is less than 100 we raise a SkyError.

The final step is to calculate the sky background value. This is achieved by first calculating the mean, median, and std of the sky region just calculated. If the median is greater than or equal to the mean, then the sky background value is just the mean, and the error in this is the std. 
If the mean is greater than the median, them we use sigma clipping, with a sigma of 3 to recalculate the mean, median, and std. We then set the sky background value to that of :math:`sky=3median\ -\ 2mean`, and the error in this to that of the recalculated std.

The function then returns the value of the sky, its error, and the FWHM's, and theta of the fitted Gaussian.

In order to get a better estimation of the sky background value, a larger image can be passed to the routine. If a larger image is passed to the routine, then the routine will use this larger image to estimate the sky background rather than the smaller cutout. The larger image size can be defined by the imganalysis.py command line option -largeimagefactor. This factor essentially makes a larger cutout image.

Below is the exposed function for calculating the sky background value.

.. automodule:: pawlikMorphLSST.skyBackground
    :members:
    :undoc-members:
    :noindex: