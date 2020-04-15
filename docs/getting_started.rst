***************
Getting Started
***************

Tutorial:

For ease of use we also provide with the package, a commandline python script to run the code, imganalysis.py.

imganalysis can be used in the commandline by running::

    python imganalysis.py

The script has several possible options:

* -h, shows the help screen.
* -f FILE, Path to a single image for analysis.
* -fo FOLDER, Path to folder of images for analysis.
* -A, runs the asymmetry calculation.
* -As, Runs the shape asymmetry calculation.
* -Aall, Runs all the implmented asymmetry calculations.
* -spm, Save calculated binary pixelmaps.
* -sci, Save cleaned image.
* -li, Use larger image cutouts to estimate sky background.
* -src, Source of the image.
* -cc, Check if any object in the provided catalogue occludes the analysed object.
* -sersic, Calculate Sersic profile.
* -fs, Size of kernel for mean filter.
* -cas, Calculate the CAS parameters (Gini, M20, r20, r80, concentation, smoothness) .

Example::

    python imganalysis.py -fo sample/data -Aall -spm -sci -src sdss -li -sersic -cas -n 4

The above command will run the code over all the SDSS images in the folder sample/data, and calculate all the asymmetry statistics, alongside the CAS and Sersic statistics.
This will generate a folder sample/output where pixelmaps of the object, cleaned images, and calculated parameters are stored.
The command will also run the code in parallel over 4 processors.