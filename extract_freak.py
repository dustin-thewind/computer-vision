# USAGE
# python extract_freak.py --image img01.jpg

# import packages
from __future__ import print_function
import argparse
import cv2
import imutils

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image, convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle OCV 2.4
if imutils.is_cv2():
    detector = cv2.FeatureDetector_create("FAST")
    extractor = cv2.DescriptorExtractor_create("FREAK")

    # init the kp detector and local invariant descriptors
    kps = detector.detect(gray)
    (kps, descs) = extractor.compute(gray, kps)

# otherwise we're using OCV 3+
else:
    detector = cv2.FastFeatureDetector_create()
    extractor = cv2.xfeatures2d.FREAK_create()

    # init the kp detector and local invariant descriptors
    kps = detector.detect(gray)
    (kps, descs) = extractor.compute(gray, kps)


# show the shape of the kp's and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps)))
print("[INFO] feature vector shape: {}".format(descs.shape))
