# USAGE
# python index_features.py -d caltech5 -f output/features.hdf5 -a 500

# import packages
from __future__ import print_function
from prod_cbir.descriptors import DetectAndDescribe
from prod_cbir.indexer import FeatureIndexer
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import random
import cv2

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help = "Path to the dataset of images to be indexed")
ap.add_argument("-f", "--features-db", required=True,
    help = "Path to where the features DB will be stored")
ap.add_argument("-a", "--approx-images", type=int, default=500,
    help = "Approximiate # of images in the dataset")
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000,
    help = "Max buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

# init the keypoint detector, local invariant descriptor
# and the descriptor pipeline
detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

# init the feaure indexer, then grab the image paths and randomly shuffle them
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],
    maxBufferSize=args["max_buffer_size"], verbose=True)
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# loop over the images in the dataset
for (i, imagePath) in enumerate(imagePaths):
    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

    # extract the filename and image calss from the image path and use to
    # construct the unique image ID
    #p = imagePath.split("/")
    #imageID = "{}:{}".format(p[-2], p[-1])

    # extract the image filename from the image path
    # then load the image itself
    #filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=320)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # describe the image
    (kps, descs) = dad.describe(image)

    # if either the keypoints ore descriptors are None, then ignore the image
    if kps is None or descs is None:
        continue

    # extract the image filename and label fromthe path, then index the features
    (label, filename) = imagePath.split("/")[-2:]
    k = "{}:{}".format(label, filename)
    fi.add(k, image.shape, kps, descs)

# finish the indexing process
fi.finish()
