if __name__ == '__main__':
    from argparse import ArgumentParser

    from pawlikMorphLSST import helpers, diagnostic, main

    parser = ArgumentParser(description="Analyse morphology of galaxies.")

    parser.add_argument("-A", action="store_true",
                        help="Calculate asymmetry parameter")
    parser.add_argument("-As", action="store_true",
                        help="Calculate shape asymmetry parameter")
    parser.add_argument("-Aall", action="store_true",
                        help="Calculate all asymmetries parameters")
    parser.add_argument("-sersic", "--sersic", action="store_true",
                        help="Calculate sersic profile.")

    parser.add_argument("-spm", "--savesegmap", action="store_true",
                        help="Save calculated binary pixelmaps.")
    parser.add_argument("-sci", "--savecleanimg", action="store_true",
                        help="Save cleaned image.")

    parser.add_argument("-li", "--largeimage", action="store_true",
                        help="Use large cutout for sky background estimation.")
    parser.add_argument("-lif", "--largeimagefactor", action="store_true",
                        help="Factor to scale cutout image size (--imgsize) so\
                        that a better background value can be estimated")

    parser.add_argument("-f", "--file", type=str, help="File which contains\
                        list of images, and RA DECS of object in image.")
    parser.add_argument("-fo", "--folder", type=str, default=None, help="Give\
                        location for where to save output files/images")
    parser.add_argument("-src", "--imgsource", type=str, default="SDSS",
                        choices=["SDSS", "LSST"], help="Telescope source\
                        of the image. Default is SDSS. This option specifies\
                        method of ingestion of the FITS image files.")
    parser.add_argument("-s", "--imgsize", type=int, default=128, help="Size of\
                        image cutout to analyse")

    parser.add_argument("-cc", "--catalogue", type=str, help="Check if any\
                         object in the provided catalogue occludes the\
                         analysed object.")
    parser.add_argument("-ns", "--numsig", type=float, default=5., help="Radial\
                        extent to which mask out stars if a catalogue is\
                        provided.")
    parser.add_argument("-fs", "--filtersize", type=int, default=3,
                        choices=[1, 3, 5, 7, 9, 11, 13, 15],
                        help="Size of kernel for mean filter")

    parser.add_argument("-par", "--parlib", type=str, default="none",
                        choices=["multi", "parsl", "none"], help="Choose which\
                        library to use to parallelise script. Default is none.")
    parser.add_argument("-n", "--cores", type=int, default=1, help="Number of\
                        cores/process to use in calculation")

    parser.add_argument("-segmap", "--segmap", action="store_true", help="If this\
                            option is provided then the script expects there\
                            to be precomputed segmentation map in the format\
                            'segmap_' + filename in the OUTPUT folder (not the same folder as used for analysis)")

    parser.add_argument("-starmask", "--starmask", action="store_true", help="If this\
                            option is provided then the script expects there\
                            to be precomputed starmasp in the format\
                            'starmask_' + filename in the OUTPUT folder (not the same folder as used for analysis)")

    parser.add_argument("-cas", "--cas", action="store_true", help="If this\
                        option is enabled, the CAS parameters are calculated\
                        (Gini, M20, r20, r80, concentation, smoothness)")
                        
    parser.add_argument("-d", "--diagnostic", action="store_true", help="If this\
                        option is enabled, diagnostic plots are made for each image")

    args = parser.parse_args()

    
    if args.segmap and args.catalogue:
        raise ValueError("Can't provide both a star catalogue and precomputed segmentation map!")

    if args.starmask and args.catalogue:
        raise ValueError("Can't provide both a star catalogue and precomputed star mask file!")

    if args.segmap and args.savesegmap:
        print("Not saving segmap as requested, because you are providing segmaps!")
        args.savesegmap = False
    
    if args.imgsource == "SDSS":
        imageInfos = helpers.getFiles(args.file)
    elif args.imgsource == "LSST":
        imageInfos = helpers.getFilesLSST(args.file, "../data/")

    outfolder = helpers.getLocation(args.folder)

    results = main.calcMorphology(imageInfos, outfolder,
                                  allAsymmetry=args.Aall,
                                  calculateSersic=args.sersic,
                                  saveSegMap=args.savesegmap,
                                  saveCleanImage=args.savecleanimg,
                                  imageSource=args.imgsource,
                                  largeImage=args.largeimage,
                                  catalogue=args.catalogue,
                                  filterSize=args.filtersize,
                                  cores=args.cores,
                                  parallelLibrary=args.parlib,
                                  numberSigmas=args.numsig,
                                  segmap=args.segmap,
                                  starMask=args.starmask,
                                  CAS=args.cas,
                                  npix=args.imgsize,
                                  largeImgFactor=args.largeimagefactor)

    if args.diagnostic:
        print(" ")
        for i in results:
            print(i.file)
            diagnostic.make_figure(i, False, save=False, show=True)
