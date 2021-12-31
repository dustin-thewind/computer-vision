# USAGE
# python search_book_covers.py --db books.csv --covers covers --query query01.png

# import packages
from __future__ import print_function
from cover_matcher import DetectAndDescribe
from cover_matcher import CoverMatcher
from imutils.feature import FeatureDetector_create, DescriptorMatcher_create
import imutils
import argparse
import glob
import csv
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True, help = "Path the book database")
ap.add_argument("-c", "--covers", required = True, help = "Path to the directory containing the book covers")
ap.add_argument("-q", "--query", required = True, help = "Path to query image")
args = vars(ap.parse_args())

# init the database dict of covers
db = {}

# loop over the db
for l in csv.reader(open(args["db"])):
    # update the db using the imageID as the key
    db[l[0]] = l[1:]

# handle if we are using OCV 2.4
if imutils.is_cv2():
    dad = DetectAndDescribe(FeatureDetector_create("SIFT"),
        DescriptorMatcher_create("SIFT"))

# otherwise we're using OCV 3+
else:
    dad = DetectAndDescribe(cv2.xfeatures2d.SIFT_create(),
        cv2.xfeatures2d.SIFT_create())

cv = CoverMatcher(dad, glob.glob(args["covers"] + "/*.png"))

# load the query image, convert it to grayscale
# and extract keypoints and descriptors
queryImage = cv2.imread(args["query"])
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = dad.describe(gray)

# try to match the book cover to a known database of images
results = cv.search(queryKps, queryDescs)

# show the query cover
cv2.imshow("query", queryImage)

# check to see if no results were found
if len(results) == 0:
    print("[INFO] no matches found!")
    cv2.waitKey(0)

# otherwise matches were found
else:
    # loop over the results
    for (i, (score, coverPath)) in enumerate(results):
        # grab the book information
        (author, title) = db[coverPath[coverPath.rfind("/") + 1:]]
        print("{}, Match confidence:{:.2f}% : Author:{} - Title:{}".format(i + 1, score * 100, author, title))

        # load the result image and show it
        result = cv2.imread(coverPath)
        cv2.imshow("result", result)
        cv2.waitKey(0)
