# ** NOTE: redis server must be running and
# redis index must exist first, to create the index run:
# python build_redis_index.py --bovw-db output/bovw.hdf5 **

# USAGE
# python evaluate_cbir.py --dataset ../ukbench --features-db output/features.hdf5 \
#	--bovw-db output/bovw.hdf5 --codebook output/vocab.cpickle --relevant ../ukbench/relevant.json

from __future__ import print_function
from prod_cbir.descriptors import DetectAndDescribe
from prod_cbir.ir import BagOfVisualWords
from prod_cbir.ir import Searcher
from prod_cbir.ir import dists
from scipy.spatial import distance
from redis import Redis
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
import numpy as np
import progressbar
import argparse
import pickle
import imutils
import json
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory if indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features DB")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to the BOVW DB")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-i", "--idf", type=str, help="Path to the inverted documented frequencies array")
ap.add_argument("-r", "--relevant", required=True, help="Path to relevant dictionary")
args = vars(ap.parse_args())

# init the keypoint detector, local invariant descriptor, descriptor pipeline
# distance metric, and inverted document frequency array
detector = FeatureDetector_create("SURF")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)
distanceMetric = dists.chi2_distance
idf = None

# if the path to the inverted document frequency array was supplied
# then load the idf array and update the distance metric
if args["idf"] is not None:
    idf = pickle.loads(open(args["idf"], "rb").read())
    distanceMetric = distance.cosine

# load the codebook vocab and init the BOVW transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

#connect to redis and init the searcher
redisDB = Redis(host="localhost", port=6379, db=0)
searcher = Searcher(redisDB, args["bovw_db"], args["features_db"], idf=idf,
    distanceMetric=distanceMetric)

# load the relevant queries dict and look up the relevant results
# for the query image
relevant = json.loads(open(args["relevant"]).read())
queryIDs = relevant.keys()

# init the accuracies and timings lists
accuracies = []
timings = []

# init the progress bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(queryIDs), widgets=widgets).start()

# loop over the images
for (i, queryID) in enumerate(sorted(queryIDs)):
    # look up the revelant results for the query image
    queryRelevant = relevant[queryID]

    # load the query image and process it
    p ="{}/{}".format(args["dataset"], queryID)
    queryImage = cv2.imread(p)
    queryImage = imutils.resize(queryImage, width=320)
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    # extract features from the query image and construct a BOVW
    (_, descs) = dad.describe(queryImage)
    hist = bovw.describe(descs).tocoo()

    # perform the search and compute the total number of relevant images
    # in the top 4 results
    sr = searcher.search(hist, numResults=4)
    results = set([r[1] for r in sr.results])
    inter = results.intersection(queryRelevant)

    # update the evaluation lists
    accuracies.append(len(inter))
    timings.append(sr.search_time)
    pbar.update(i)

# release any pointers allocated by the searcher
searcher.finish()
pbar.finish()

# show eval information to user
accuracies = np.array(accuracies)
timings = np.array(timings)
print("[INFO] Accuracy: Mean(u)={:.2f}, Standard Deviation(o)={:.2f}".format(accuracies.mean(), accuracies.std()))
print("[INFO] Timings: Mean(u)={:.2f}, Standard Deviation(o)={:.2f}".format(timings.mean(), timings.std()))
