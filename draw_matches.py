# USAGE
# python draw_matches.py --first jp_01.png --second jp_02.png --detector SURF --extractor SIFT

# import packages
from __future__ import print_function
import numpy as np
import cv2
import argparse
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="Path to the first image")
ap.add_argument("-s", "--second", required=True, help="Path to the second image")
ap.add_argument("-d", "--detector", type=str, default="SURF",
    help="Keypoint detector to use."
        "Options ['BRISK', 'DENSE', 'DOG', 'SIFT', 'FAST', 'FASTHESSIAN', 'SURF', 'GFTT', 'HARRIS', 'MSER', 'ORB', 'STAR']")
ap.add_argument("-e", "--extractor", type=str, default="SIFT",
    help="Feature extractor to use. Options['RootSIFT', 'SIFT', 'SURF']")
ap.add_argument("-m", "--matcher", type=str, default="BruteForce",
    help="Feature matcher to use. Options['BruteForce', 'BruteForce-Hamming', 'BruteForce-SL2', 'BruteForce-L1', 'FlannBased']")
ap.add_argument("-v", "--visualize", type=str, default="Yes",
    help="Whether the visualization image should be shown. Options['Yes', 'No', 'Each']")
args = vars(ap.parse_args())

# init the feature detector
# if the user entered detector as 'DOG' or 'FASTHESSIAN' use the correct value
if args["detector"] == "DOG":
    detector = FeatureDetector_create("SIFT")
elif args["detector"] == "SURF":
    detector = FeatureDetector_create("SURF")
else:
    detector = FeatureDetector_create(args["detector"])

# init the feature extractor
extractor = DescriptorExtractor_create(args["extractor"])

# init the keypoint matcher
matcher = DescriptorMatcher_create(args["matcher"])

# load the two images and conver to grayscale
imgA = cv2.imread(args["first"])
imgB = cv2.imread(args["second"])
grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

# detect keypoints in the two images
kpsA = detector.detect(grayA)
kpsB = detector.detect(grayB)

# extract features from each of the keypoint regions in the images
(kpsA, featuresA) = extractor.compute(grayA, kpsA)
(kpsB, featuresB) = extractor.compute(grayB, kpsB)

# match the keypoints using the Euclidean distance and
# init the list of actual matches
rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
matches = []

if rawMatches is not None:
    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance passes David Lowe's ratio test
        if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # show some diag info
    print("[INFO] # of keypoints from first image: {}".format(len(kpsA)))
    print("[INFO] # of keypoints from second image: {}".format(len(kpsB)))
    print("[INFO] # matched keypoints: {}".format(len(matches)))

    # init the output visualization image
    (hA, wA) = imgA.shape[:2]
    (hB, wB) = imgB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imgA
    vis[0:hB, wA:] = imgB

    # loop over the matches
    for (trainIdx, queryIdx) in matches:
        # generate a random color and draw the match
        color = np.random.randint(0, high=255, size=(3,))
        color = tuple(map(int, color))
        ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
        ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
        cv2.line(vis, ptA, ptB, color, 2)

        # check to see if each match should be visualized individually
        if args["visualize"] == "Each":
            cv2.imshow("Matched", vis)
            print("[INFO] feature vector shape: {}".format(np.shape(matches)))
            cv2.waitKey(0)

    # show the visualization
    if args["visualize"] == "Yes":
        cv2.imshow("Matched", vis)
        print("[INFO] feature vector shape: {}".format(np.shape(matches)))
        cv2.waitKey(0)
