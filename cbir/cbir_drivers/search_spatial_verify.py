# ** NOTE: redis server must be running and
# redis index must exist first, to create the index run:
# python build_redis_index.py --bovw-db output/bovw.hdf5 **

# USAGE

# python search_spatial_verify.py --dataset ../ukbench --features-db output/features.hdf5 --bovw-db output/bovw.hdf5 \
#   --codebook output/vocab.cpickle --relevant ../ukbench/relevant.json --query ../ukbench/ukbench00258.jpg

# import packages
from __future__ import print_function
from prod_cbir.descriptors import DetectAndDescribe
from prod_cbir.ir import BagOfVisualWords
from prod_cbir.ir import SpatialVerifier
from prod_cbir.ir import Searcher
from ...python import ResultsMontage
from scipy.spatial import distance
from redis import redis
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
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
ap.add_argument("-q", "--query", required=True, help="Path to query image")
args = vars(ap.parse_args())

# init the keypoint detector, local invariant descriptor, descriptor pipeline
# distance metric, and inverted document frequency array
detector = FeatureDetector_create("SURF")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

# load the IDF array and codebook vocab, then
# init the BOVW transformer
idf = pickle.loads(open(args["idf"], "rb").read())
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# load the relevant queries dict and look up the relevant results
# for the query image
relevant = json.loads(open(args["relevant"]).read())
queryFilename = args["query"][args["query"].rfind("/") + 1:]
queryRelevant = relevant[queryFilename]

# load the query image and process it
queryImage = cv2.imread(args["query"])
cv2.imshow("query", imutils.resize(queryImage, width=320))
queryImage = imutils.resize(queryImage, width=320)
queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

# extract features from the query image and construct a bovw from it
(queryKps, queryDescs) = dad.describe(queryImage)
queryHist = bovw.describe(queryDescs).tocoo()

# connect to redis and perform the search
redisDB = Redis(host="localhost", port=6379, db=0)
searcher = Searcher(redisDB, args["bovw_db"], args["features_db"], idf=idf,
    distanceMetric=distanceMetric)
sr = searcher.search(hist, numResults=20)
print("[INFO] search took: {:.2f}s".format(sr.search_time))

# spatially verify the results
SpatialVerifier = SpatialVerifier(args["features_db"], idf, vocab)
sv = SpatialVerifier.rerank(queryKps, queryDescs, sr, numResults=20)
print("[INFO] spatial verification took: {:2f}s".format(sv.search_time))

# init the results montage
montage = ResultsMontage((240, 320), 5, 20)

# loop over the individual results
for (i, (score, resultID, resultIdx)) in enumerate(sv.results):
    # load the result image and display it
    print("[RESULT] {result_num}. {result} - {score:.2f}".format(result_num=i + 1,
        result=resultID, score=score))
    result = cv2.imread("{}/{}".format(args["dataset"], resultID))
    montage.addResult(result, text="#{}".format(i + 1),
        highlight=resultID in queryRelevant)

# show the output of the results
cv2.imshow("results", imutils.resize(montage.montage, height=700))
cv2.waitKey(0)
searcher.finish()
SpatialVerifier.finish
