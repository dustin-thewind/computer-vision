# USAGE
# python extract_rootsift.py -i /image1.png

# import packages
from __future__ import print_function
from rootsift import RootSIFT
import cv2
import imutils
import argparse

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we are detecting and drawing DoG KP's with OCV 2.4
if imutils.is_cv2():
    detector = cv2.FeatureDetector_create("SIFT")
    extractor = cv2.DescriptorExtractor_create("SIFT")

    # detect keypoints, and then extract local invariant desciptors
    kps = detector.detect(gray)

# otherwise we're using OCV 3+
else:
    detector = cv2.xfeatures2d.SIFT_create()
    extractor = RootSIFT()

    # detect keypoints
    (kps, _) = detector.detectAndCompute(gray, None)

# extract local invariant desciptors
(kps, descs) = extractor.compute(gray, kps)

# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps)))
print("[INFO] feature vector shape: {}".format(descs.shape))
