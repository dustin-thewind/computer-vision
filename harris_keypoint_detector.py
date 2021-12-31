# import packages
from __future__ import print_function
import numpy as np
import cv2
import imutils
import argparse

def harris(gray, blockSize=2, apertureSize=3, k=0.1, T=0.02):
    # convert input image to a floating point data type
    # then compute the harris corner matrix
    gray = np.float32 (gray)
    H = cv2.cornerHarris(gray, blockSize, apertureSize, k)

    # for every x,y coordinate where the Harris value is above
    # the threshold, create a keypoint
    # the harris detector returns a keypoint with 3px radius size
    kps = np.argwhere(H > T * H.max())
    kps = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kps]

    # return the harris keypoints
    return kps

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we are detecting harris KP's with OCV 2.4
if imutils.is_cv2():
    detector = cv2.FeatureDetector_create("HARRIS")
    kps = detector.detect(gray)

# otherwise we're using OCV 3+
else:
    kps = harris(gray)

# print out how many kps there are
print("num kps: {}".format(len(kps)))

# loop over the kps and draw them
for kp in kps:
    r = int(0.5 * kp.size)
    (x, y) = np.int0(kp.pt)
    cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# show the image
cv2.imshow("images", np.hstack([orig, image]))
cv2.waitKey(0)
