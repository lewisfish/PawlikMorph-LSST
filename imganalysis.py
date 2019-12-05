if __name__ == '__main__':
    from argparse import ArgumentParser

    from pawlikMorphLSST import helpers, diagnostic

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
    parser.add_argument("-src", "--imgsource", type=str, default="sdss", choices=["sdss", "hsc"],
                        help="Source of the image.")
    parser.add_argument("-cc", "--catalogue", type=str, help="Check if any object in the\
                        provided catalogue occludes the analysed object.")

    args = parser.parse_args()

    files = helpers.getFiles(args.imgsource, file=args.file, folder=args.folder)
    curfolder, outfolder = helpers.getLocation(file=args.file, folder=args.folder)
    results = helpers.calcMorphology(files, outfolder, allAsymmetry=args.Aall,
                                     savePixelMap=args.savepixmap,
                                     saveCleanImage=args.savecleanimg,
                                     imageSource=args.imgsource,
                                     largeImage=args.largeimage,
                                     catalogue=args.catalogue)
    print(" ")
    for i in results:
        print(i.file)
        diagnostic.make_figure(i)
