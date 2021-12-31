# USAGE
# python recognize_license_plates.py --images ../testing_lp_dataset
#   -c simple_char.cpickle -d simple_digit.cpickle
# import packages
from __future__ import print_function
from anpr.license_plate import LicensePlateDetector
from anpr.descriptors import BlockBinaryPixelSum
from imutils import paths
import numpy as np
import pickle
import argparse
import imutils
import cv2

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True, help = "Path to the images to be classified")
ap.add_argument("-c", "--char-classifier", required = True, help = "Path to the output char classifier")
ap.add_argument("-d", "--digit-classifier", required = True, help = "Path to the output digit classifier")
args = vars(ap.parse_args())

charModel = pickle.loads(open(args["char_classifier"], "rb").read())
digitModel = pickle.loads(open(args["digit_classifier"], "rb").read())

# init the descriptor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# loop over the images
for imagePath in sorted(list(paths.list_images(args["images"]))):
    # load the image
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)

    # if the width is >640 px then resize
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    # init the license plate detector, detect the license plates and characters
    lpd = LicensePlateDetector(image)
    plates = lpd.detect()

    # loop over the detected plates
    for (lpBox, chars) in plates:
        # restructure lpbox
        lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)

        # init the text containing the recognized characters
        text = ""

        for (i, char) in enumerate(chars):
            # show the char
            char = LicensePlateDetector.preprocessChar(char)
            if char is None:
                continue
            features = desc.describe(char).reshape(1, -1)

            if i < 3:
                prediction = charModel.predict(features)[0]

            # otherwise, use the digit classifier
            else:
                prediction = digitModel.predict(features)[0]

            # update the text of recognized characters
            text += prediction.upper()

        # only draw the characters and bounding box if there characters we can display
        if len(chars) > 0:
            # compute the center of the license plate bounding box
            M = cv2.moments(lpBox)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw the license plate region and license palte text on the image
            cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)
            cv2.putText(image, text, (cX - (cX // 5), cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2)

        # display the output images
    cv2.imshow("image {}".format(filename), image)
    cv2.waitKey(0)
