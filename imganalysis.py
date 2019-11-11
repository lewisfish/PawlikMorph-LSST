if __name__ == '__main__':
    import csv
    import sys
    import time
    import warnings
    from argparse import ArgumentParser
    from pathlib import Path

    from astropy.io import fits
    from astropy.utils.exceptions import AstropyWarning
    import numpy as np
    import matplotlib.pyplot as plt

    from pawlikMorphLSST import asymmetry, apertures, pixmap, imageutils

    # from apertures import makeaperpixmaps, distarr, aperpixmap
    # from asymmetry import calcA, minapix
    # from imageutils import skybgr, cleanimg, cutoutImg
    # from pixmap import pixelmap

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
    parser.add_argument("-src", "--imgsource", type=str, choices=["sdss", "hsc"],
                        help="Source of the image.")

    args = parser.parse_args()

    if not args.file and not args.folder:
        print("Script needs input images to work!!")
        sys.exit()

    # get image size. Assume all images the same size and are square
    # suppress warnings about unrecognised keywords
    warnings.simplefilter('ignore', category=AstropyWarning)
    # add files to a list
    files = []
    if args.file:
        files.append(Path(args.file))
    elif args.folder:
        # TODO change to a generator
        files = list(Path(args.folder).glob(f"{args.imgsource}cutout*.fits"))
    if files[0].exists():
        data = fits.getdata(files[0])
        imgsize = data.shape[0]
    else:
        print(f"Fits image:{files[0].name} does not exist!")
        sys.exit()

    if args.folder:
        outfolder = Path(args.folder).parents[0] / "output/"
    else:
        outfolder = Path(args.file).parents[0] / "output/"

    if not outfolder.exists():
        outfolder.mkdir()

    outfile = outfolder / "parameters.csv"
    csvfile = open(outfile, mode="w")
    paramwriter = csv.writer(csvfile, delimiter=",")
    paramwriter.writerow(["file", "apix", "r_max", "sky", "sky_err", "A", "Abgr",
                          "As", "As90", "time"])

    files.sort()
    for file in files:
        # set default values for calculated parameters
        apix = (-99, -99)
        r_max = -99
        sky = -99
        sky_err = -99
        A = [-99, -99]
        As = [-99, -99]
        As90 = [-99, -99]

        s = time.time()

        if not file.exists():
            print(f"Fits image:{file.name} does not exist!")
            continue
        print(file)

        data = fits.getdata(file)
        imgsize = data.shape[0]
        # The following is required as fits files are big endian and skimage
        # assumes little endian.
        # https://stackoverflow.com/a/30284033/6106938
        # https://en.wikipedia.org/wiki/Endianness
        data = data.byteswap().newbyteorder()

        if not data.shape[0] == data.shape[1]:
            print("ERROR: wrong image size. Please preprocess data!")
            sys.exit()

        # get sky background value and error
        if args.largeimage:
            filename = file.name
            if args.imgsource == "sdss":
                filename = filename.replace("sdss", "sdssl", 1)
            elif args.imgsource == "hsc":
                filename = filename.replace("hsc", "hscl", 1)

            infile = Path(args.folder) / Path(filename)
            if infile.exists():
                datatmp = fits.getdata(Path(args.folder) / Path(filename))  # FIXME: crashes if given a single file
                sky, sky_err, flag = imageutils.skybgr(datatmp, datatmp.shape[0], data)
            else:
                print(f"{infile} does not exist!")
                sky, sky_err, flag = imageutils.skybgr(data, imgsize)
        else:
            sky, sky_err, flag = imageutils.skybgr(data, imgsize)

        if flag != 0:
            if flag == 1:
                print(f"ERROR! Skybgr not calculated for {file} as skyregion is less than 100 pixels.")
            else:
                print(f"ERROR! Skybgr not calculated for {file} as Gaussian could not be fitted to image.")
            paramwriter.writerow([f"{file}", f"0", f"0", f"0", f"0", f"0", f"0", f"0", f"0", f"0"])
            print(" ")
            continue

        mask = pixmap.pixelmap(data, sky + sky_err, 3)
        data -= sky

        # clean image of external sources
        data = imageutils.cleanimg(data, mask)
        if args.savecleanimg:
            filename = file.name
            filename = "clean_" + filename
            outfile = outfolder / filename
            fits.writeto(outfile, data, overwrite=True)

        if args.savepixmap:
            filename = file.name
            filename = "pixelmap_" + filename
            outfile = outfolder / filename
            fits.writeto(outfile, mask, overwrite=True)

        objectpix = np.nonzero(mask == 1)
        cenpix = np.array([int(imgsize/2) + 1, int(imgsize/2) + 1])

        distarray = apertures.distarr(imgsize, imgsize, cenpix)
        objectdist = distarray[objectpix]
        r_max = np.max(objectdist)
        aperturepixmap = apertures.aperpixmap(imgsize, r_max, 9, 0.1)

        apix = asymmetry.minapix(data, mask, aperturepixmap)
        angle = 180.

        if args.A or args.Aall:
            A = asymmetry.calcA(data, mask, aperturepixmap, apix, angle, noisecorrect=True)

        if args.As or args.Aall:
            As = asymmetry.calcA(mask, mask, aperturepixmap, apix, angle)
            As90 = asymmetry.calcA(mask, mask, aperturepixmap, apix, 90.)

        f = time.time()
        timetaken = f - s
        paramwriter.writerow([f"{file}", f"{apix}", f"{r_max}", f"{sky}", f"{sky_err}", f"{A[0]}", f"{A[1]}", f"{As[0]}", f"{As90[0]}", f"{timetaken}"])

    print(" ")
    csvfile.close()
