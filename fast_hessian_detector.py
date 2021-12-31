# import packages
from __future__ import print_function
import numpy as np
import cv2
import imutils
import argparse

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we are detecting and drawing DoG KP's with OCV 2.4
if imutils.is_cv2():
    detector = cv2.FeatureDetector_create("SURF")
    kps = detector.detect(gray)

# otherwise we're using OCV 3+
else:
    detector = cv2.xfeatures2d.SURF_create()
    (kps, _) = detector.detectAndCompute(gray, None)

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
