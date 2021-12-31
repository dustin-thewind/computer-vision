# USAGE
# python extract_bovw.py --features-db output/features.hdf5 --codebook output/vocab.cpickle \
#       --bovw-db output/bovw.hdf5 --idf output/idf.cpickle

# import packages
from prod_cbir.ir import BagOfVisualWords
from prod_cbir.indexer import BOVWIndexer
from prod_cbir.descriptors import PBOW
import argparse
import pickle
import h5py

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True,
    help = "Path to the features DB")
ap.add_argument("-c", "--codebook", required=True,
    help = "Path to the codebook")
ap.add_argument("-p", "--pbow-db", required=True,
    help = "Path to where the PBOW DB will be stored")
ap.add_argument("-s", "--max-buffer-size", type=int, default=500,
    help = "Max buffer size for # of features to store in memory")
ap.add_argument("-l", "--levels", type=int, default=2,
    help = "# of pyramid levels to generate")
args = vars(ap.parse_args())

# load the codebook vocab and itit the BOVW transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)
pbow = PBOW(bovw, numLevels=args["levels"])

# open the features DB and init the BOVW indexer
featureDim = PBOW.featureDim(bovw.codebook.shape[0], args["levels"])
featuresDB = h5py.File(args["features_db"], mode="r")
bi = BOVWIndexer(featureDim, args["pbow_db"], estNumImages=featuresDB["image_ids"].shape[0],
    maxBufferSize=args["max_buffer_size"])

# loop over the image ID's and index
for (i, imageID) in enumerate(featuresDB["image_ids"]):
    # grab the image dimensions, along with the index lookup
    # values from the DB
    (h, w) = featuresDB["image_dims"][i]
    (start, end) = featuresDB["index"][i]

    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        bi._debug("processed {} images".format(i), msgType="[PROGRESS]")

    # extract the keypoints and feature vectors for the current image
    # using the starting and ending offsets while ignoring the keypoints
    # and then create the PBOW representation
    kps = featuresDB["features"][start:end][:, :2]
    descs = featuresDB["features"][start:end][:, 2:]
    hist = pbow.describe(w, h, kps, descs)

    # add the BOVW to the index
    bi.add(hist)

# close the features DB and finish indexing
featuresDB.close()
bi.finish()
