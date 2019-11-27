if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    from pawlikMorphLSST import helpers

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

    files = helpers.getFiles(args)
    curfolder, outfolder = helpers.getLocation(args)
    helpers.calcMorphology(files, outfolder, args)
