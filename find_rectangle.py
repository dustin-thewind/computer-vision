#usage 'python find_rectangle.py --dataset output'

#import packages
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import argparse
import glob
import cv2
import imutils

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help = "Path to the dataset directory")
args = vars(ap.parse_args())

#grab the image paths from disk and init the data matrix
imagePaths = sorted(glob.glob(args["dataset"] + "/*.jpg"))
data = []

#loop over the images in the dataset dir
for imagePath in imagePaths:
    #load the image, convert it to grayscale, threshhold it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

    #find the contours in the image, keep only the largest
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    #extract the ROI from the image, resize it to canonincal size
    #compute the hu moments feature vector for the ROI
    #update the data matrix
    (x, y, w, h) = cv2.boundingRect(c)
    roi = cv2.resize(thresh[y:y + h, x:x + w], (50, 50))
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()
    data.append(moments)

#compute the distance between all entries in the data matrix
#take the sum of the distances for each row, grab the
#row with the largest distance
D = pairwise_distances(data).sum(axis=1)
i = np.argmax(D)

#display the outlier image
image = cv2.imread(imagePaths[i])
cv2.imshow("outlier", image)
cv2.waitKey(0)
