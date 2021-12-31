# import packages
from __future__ import print_function
import numpy as np
import cv2
import imutils
import argparse

def gftt(gray, maxCorners=0, qualityLevel=0.01, minDistance=1,
    mask=None, blockSize=3, useHarrisDetector=False, k=0.04):
    # compute GFTT keypoints using the supplied paramaters (in OCV3+ only)
    kps = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel,
        minDistance, mask=mask, blockSize=blockSize,
        useHarrisDetector=useHarrisDetector, k=k)

    # create and return 'KeyPoint' objects
    return [cv2.KeyPoint(pt[0][0], pt[0][1], 3) for pt in kps]

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we are detecting and drawing GFTT KP's with OCV 2.4
if imutils.is_cv2():
    detector = cv2.FeatureDetector_create("GFTT")
    kps = detector.detect(gray)

# otherwise we're using OCV 3+
else:
    kps = gftt(gray)

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
