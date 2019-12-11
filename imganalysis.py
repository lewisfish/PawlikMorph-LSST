if __name__ == '__main__':
    from argparse import ArgumentParser

    from pawlikMorphLSST import helpers, diagnostic, morphology

    parser = ArgumentParser(description="Analyse morphology of galaxies.")

    parser.add_argument("-f", "--file", type=str,
                        help="Path to single image to be analysed")
    parser.add_argument("-fo", "--folder", type=str,
                        help="Path to folder where images are to be analysed")
    parser.add_argument("-A", action="store_true",
                        help="Calculate asymmetry parameter")
    parser.add_argument("-As", action="store_true",
                        help="Calculate shape asymmetry parameter")
    parser.add_argument("-Aall", action="store_true",
                        help="Calculate all asymmetries parameters")
    parser.add_argument("-spm", "--savepixmap", action="store_true",
                        help="Save calculated binary pixelmaps.")
    parser.add_argument("-sci", "--savecleanimg", action="store_true",
                        help="Save cleaned image.")
    parser.add_argument("-li", "--largeimage", action="store_true",
                        help="Use large cutout for sky background estimation.")
    parser.add_argument("-src", "--imgsource", type=str, default="sdss",
                        choices=["sdss", "hsc"], help="Source of the image.")
    parser.add_argument("-cc", "--catalogue", type=str, help="Check if any\
                         object in the provided catalogue occludes the\
                         analysed object.")
    parser.add_argument("-sersic", "--sersic", action="store_true",
                        help="Calculate sersic profile.")
    parser.add_argument("-fs", "--filtersize", type=int, default=3,
                        choices=[1, 3, 5, 7, 9, 11, 13, 15],
                        help="Size of kernel for mean filter")
    parser.add_argument("-n", "--cores", type=int, help="Number of\
                        cores/process to use in calculation")

    args = parser.parse_args()

    files = helpers.getFiles(args.imgsource, file=args.file,
                             folder=args.folder)

    curfolder, outfolder = helpers.getLocation(file=args.file,
                                               folder=args.folder)

    results = morphology.calcMorphology(files, outfolder,
                                        allAsymmetry=args.Aall,
                                        calculateSersic=args.sersic,
                                        savePixelMap=args.savepixmap,
                                        saveCleanImage=args.savecleanimg,
                                        imageSource=args.imgsource,
                                        largeImage=args.largeimage,
                                        catalogue=args.catalogue,
                                        filterSize=args.filtersize,
                                        cores=args.cores)

    print(" ")
    for i in results:
        print(i.file)
        diagnostic.make_figure(i, save=True)
