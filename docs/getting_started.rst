***************
Getting Started
***************

Tutorial:

For ease of use we also provide with the package, a commandline python script to run the code, imganalysis.py.

imganalysis can be used in the commandline by running::

    python imganalysis.py

The script has several possible options:

  -h, --help            show this help message and exit.
  -A                    Calculate asymmetry parameter.
  -As                   Calculate shape asymmetry parameter.
  -Aall                 Calculate all asymmetries parameters.
  -sersic, --sersic     Calculate Sersic profile.
  -spm, --savepixmap    Save calculated binary pixelmaps.
  -sci, --savecleanimg  Save cleaned image.
  -li, --largeimage     Use large cutout for sky background estimation.
  -lif, --largeimagefactor
                        Factor to scale cutout image size (--imgsize) so that a better background value can be estimated.
  -f FILE, --file FILE  File which contains list of images, and RA DECS of object in image.
  -fo FOLDER, --folder FOLDER
                        Give location for where to save output files/images
  -src {SDSS,LSST}, --imgsource {SDSS,LSST}
                        Telescope source of the image. Default is SDSS. This option specifies method of ingestion of the FITS image files.
  -s IMGSIZE, --imgsize IMGSIZE
                        Size of image cutout to analyse.
  -cc CATALOGUE, --catalogue CATALOGUE
                        Check if any object in the provided catalogue occludes the analysed object.
  -ns NUMSIG, --numsig NUMSIG
                        Radial extent to which mask out stars if a catalogue is provided.
  -fs {1,3,5,7,9,11,13,15}, --filtersize {1,3,5,7,9,11,13,15}
                        Size of kernel for mean filter.
  -par {multi,parsl,none}, --parlib {multi,parsl,none}
                        Choose which library to use to parallelise script. Default is none.
  -n CORES, --cores CORES
                        Number of cores/process to use in calculation
  -m, --mask            If this option is provided then the script expects there to be precomputed masks in the format pixelmap_ + filename in the same folder as the images for analysis
  -cas, --cas           If this option is enabled, the CAS parameters are calculated (Gini, M20, r20, r80, concentation, smoothness)


Example::

    python imganalysis.py -f images.csv -fo output/ -Aall -spm -sci -src sdss -li -sersic -cas -par multi -n 4

The above command will run the code over all the images in the file images.csv, and calculate all the asymmetry statistics, alongside the CAS and Sersic statistics.
This will generate a folder output where pixelmaps of the object, cleaned images, and calculated parameters (parameters.csv) are stored.
The command will also run the code in parallel over 4 processors.

If imganalysis.py is run with the -spm and -sci options, then it will automatically plot the various outputs using matplotlib at the end of a run. This can be useful to asses by eye the various setting used in that run for generating the segmentation map, and size of star to mask out.

imganalysis.py is provided as an easy option for calculating various morphology statistics.
However, the package can also be used as a general purpose library for building your own scripts. The package publicly exposes several of the functions in order to achieve this.