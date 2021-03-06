# USAGE
# python gather_examples.py --images ../full_lp_dataset --examples output/examples


# import packages
from __future__ import print_function
from anpr.license_plate import LicensePlateDetector
from imutils import paths
import traceback
import argparse
import imutils
import numpy as np
import random
import cv2
import os

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help = "path to the images to be classified")
ap.add_argument("-e", "--examples", required=True, help = "path to the output examples directory")
args = vars(ap.parse_args())

# randomly select a portion of the images and init the dictionary of character counts
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:int(len(imagePaths) * 0.5)]
counts = {}

# loop over the images
for imagePath in imagePaths:
    # show the image path
    print("[EXAMINING] {}".format(imagePath))

    try:
        # load the image
        image = cv2.imread(imagePath)

        # if width is greater than 640 pixels, then resize it
        if image.shape[1] > 640:
            image = imutils.resize(image, width=640)

        # init the LPD and detect characters on the license plate
        lpd = LicensePlateDetector(image, numChars=7)
        plates = lpd.detect()

        # loop over the license plates
        for (lpBox, chars) in plates:
            # restructure lpBox
            lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)

            # draw the bounding box surrounding the license plate and display it for
            # reference purposes
            plate = image.copy()
            cv2.drawContours(plate, [lpBox], -1, (0, 255, 0), 2)
            cv2.imshow("License Plate", plate)

            # loop over the characters
            for char in chars:
                # dissplay the char and wait for a key press
                cv2.imshow("char", char)
                key = cv2.waitKey(0)

                # if they '`' key was pressed, then ignore the char
                if key == ord("`"):
                    print("[IGNORING] {}".format(imagePath))
                    continue

                # grab the key that was pressed and contruct the path to the output dir
                key = chr(key).upper()
                dirPath = "{}/{}".format(args["examples"], key)

                # if the output directory doesn't exist, create it
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)

                # write the labeled character to file
                count = counts.get(key, 1)
                path = "{}/{}.png".format(dirPath, str(count).zfill(5))
                cv2.imwrite(path, char)

                # increment the count for the current key
                counts[key] = count + 1

    # we are trying to ctrl-c out of the script, so break the loop
    except KeyboardInterrupt:
        break

    # an uknown error occurred for this particular image, so do not
    # process it and display a traceback for debugging purposes
    except:
        print(traceback.format_exc())
        print("[ERROR] {}".format(imagePath))
