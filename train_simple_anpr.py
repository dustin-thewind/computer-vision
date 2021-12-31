# USAGE
# python train_model_classifier.py -d caltech5 -f output/features_caltech5.hdf5 /
#   -b output/bovw_caltech5.hdf5 -m output/model.cpickle

# import packages
from __future__ import print_function
from anpr.descriptors import BlockBinaryPixelSum
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import pickle
import cv2
import imutils

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--fonts", required=True,
    help = "Path to the fonts dataset")
ap.add_argument("-c", "--char-classifier", required=True,
    help = "Path to the output char classifier")
ap.add_argument("-d", "--digit-classifier", required=True,
    help = "Path to the output digit classifier")
args = vars(ap.parse_args())

# init the chars string
engAlphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

# init the data and labels lists for the alphabet and digits
alphabetData = []
digitsData = []
alphabetLabels = []
digitsLabels = []

# init the descriptor
print("[INFO] describing font examples...")
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# loop over the font paths
for fontPath in paths.list_images(args["fonts"]):
    # load the font image, convert it to grayscale, and treshold it
    font = cv2.imread(fontPath)
    font = cv2.cvtColor(font, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(font, 128, 255, cv2.THRESH_BINARY_INV)[1]

    # detect contours in the thresholded image, and sort them left to right
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c:(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1]))

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # grab the bounding box for the contour, extract the ROI, and extract features
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        features = desc.describe(roi)

        # check to see if this is an alphabet char
        if i < 26:
            alphabetData.append(features)
            alphabetLabels.append(engAlphabet[i])

        # otherwise this is a digit
        else:
            digitsData.append(features)
            digitsLabels.append(engAlphabet[i])

# train the character classifier
print("[INFO] fitting character model...")
charModel = LinearSVC(C=1.0, random_state=42)
charModel.fit(alphabetData, alphabetLabels)

# train the digit classifier
print("[INFO] fitting digit model...")
digitModel = LinearSVC(C=1.0, random_state=42)
digitModel.fit(digitsData, digitsLabels)

# dump the character classifier to file
print("[INFO] dumping character model...")
f = open(args["char_classifier"], "wb")
f.write(pickle.dumps(charModel))
f.close()

# dump the digit classifier to file
print("[INFO] dumping digit model...")
f = open(args["digit_classifier"], "wb")
f.write(pickle.dumps(digitModel))
f.close()
