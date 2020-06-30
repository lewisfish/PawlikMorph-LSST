************
ObjectMasker
************

This module is only used if a star catalogue is used via the command line option --cc.
The catalogue is expected to be a comma separated CSV file with the following header and columns:

```objID, ra, dec, psfMag_r, type```

Where ObjId is the object ID, RA is the right ascension, DEC is the Declination, psfMAG_r is the PSF magnitude for the r band, and type is the object type from {STAR, GALAXY, COSMICRAY, UNKNOWN}.

The function simply reads in the star catalogue and checks for which object to match with. By default this is only STAR, however the other types can be enabled.
The function then converts RA DEC pairs to pixel coordinates and uses these to check if the object is in the image. If the object is in the image it then checks to see if occludes the segmentation map.

.. automodule:: pawlikMorphLSST.objectMasker
    :members:
    :undoc-members:
    :noindex: