if __name__ == '__main__':
    import csv
    import sys
    import time
    import warnings
    from argparse import ArgumentParser
    from pathlib import Path

    from astropy.io import fits
    from astropy.utils.exceptions import AstropyWarning
    from astropy import wcs

    import numpy as np
    # import matplotlib.pyplot as plt

    from pawlikMorphLSST import asymmetry, apertures, pixmap, imageutils, objectMasker, helpers

    # suppress warnings about unrecognised keywords
    # warnings.simplefilter('ignore', category=AstropyWarning)

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

    outfile = outfolder / "parameters.csv"
    csvfile = open(outfile, mode="w")
    paramwriter = csv.writer(csvfile, delimiter=",")
    paramwriter.writerow(["file", "apix", "r_max", "sky", "sky_err", "A", "Abgr",
                          "As", "As90", "time", "star_flag"])

    if args.catalogue:
        outfile = outfolder / "occluded-object-locations.csv"
        objcsvfile = open(outfile, mode="w")
        objwriter = csv.writer(objcsvfile, delimiter=",")
        objwriter.writerow(["file", "ra", "dec", "type"])

    for file in files:

        try:
            img, header, imgsize = helpers.checkFile(file)
        except IOError:
            print(f"File {file}, does not exist!")
            continue
        except AttributeError as e:
            continue

        # set default values for calculated parameters
        apix = (-99, -99)
        r_max = -99
        sky = -99
        sky_err = -99
        A = [-99, -99]
        As = [-99, -99]
        As90 = [-99, -99]

        s = time.time()

        print(file)

        # get sky background value and error
        sky, sky_err, flag = imageutils.skybgr(img, imgsize, file, args)

        if flag != 0:
            if flag == 1:
                print(f"ERROR! Skybgr not calculated for {file} as skyregion is less than 100 pixels.")
            else:
                print(f"ERROR! Skybgr not calculated for {file} as Gaussian could not be fitted to image.")
            paramwriter.writerow([f"{file}", f"0", f"0", f"0", f"0", f"0", f"0", f"0", f"0", f"0", "False"])
            filename = file.name
            filename = "pixelmap_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=np.zeros_like(img), header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')
            print(" ")
            continue

        # img = imageutils.maskstarsSEG(img)
        mask = pixmap.pixelmap(img, sky + sky_err, 3)

        star_flag = False
        if args.catalogue:
            w = wcs.WCS(header)
            star_flag, objlist = objectMasker.objectOccluded(mask, file.name, args.catalogue, w, galaxy=True, cosmicray=True, unknown=True)
            if star_flag:
                for i, obj in enumerate(objlist):
                    if i == 0:
                        objwriter.writerow([f"{file}", obj[0], obj[1], obj[2]])
                    else:
                        objwriter.writerow(["", obj[0], obj[1], obj[2]])

        img -= sky

        # clean image of external sources
        img = imageutils.cleanimg(img, mask)
        if args.savecleanimg:
            filename = file.name
            filename = "clean_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=img, header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')

        if args.savepixmap:
            filename = file.name
            filename = "pixelmap_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=mask, header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')

        objectpix = np.nonzero(mask == 1)
        cenpix = np.array([int(imgsize/2) + 1, int(imgsize/2) + 1])

        distarray = apertures.distarr(imgsize, imgsize, cenpix)
        objectdist = distarray[objectpix]
        r_max = np.max(objectdist)
        aperturepixmap = apertures.aperpixmap(imgsize, r_max, 9, 0.1)

        apix = asymmetry.minapix(img, mask, aperturepixmap)
        angle = 180.

        if args.A or args.Aall:
            A = asymmetry.calcA(img, mask, aperturepixmap, apix, angle, noisecorrect=True)

        if args.As or args.Aall:
            As = asymmetry.calcA(mask, mask, aperturepixmap, apix, angle)
            As90 = asymmetry.calcA(mask, mask, aperturepixmap, apix, 90.)

        f = time.time()
        timetaken = f - s
        paramwriter.writerow([f"{file}", f"{apix}", f"{r_max}", f"{sky}", f"{sky_err}", f"{A[0]}", f"{A[1]}", f"{As[0]}", f"{As90[0]}", f"{timetaken}", f"{star_flag}"])

    if args.catalogue:
        objcsvfile.close()
    csvfile.close()
