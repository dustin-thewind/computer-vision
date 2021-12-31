# USAGE
# python extract_bovw.py --features-db output/features.hdf5 --codebook output/vocab.cpickle \
#       --bovw-db output/bovw.hdf5 --idf output/idf.cpickle

# import packages
from prod_cbir.ir import BagOfVisualWords
from prod_cbir.indexer import BOVWIndexer
import argparse
import pickle
import h5py

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True,
    help = "Path to the features DB")
ap.add_argument("-c", "--codebook", required=True,
    help = "Path to the codebook")
ap.add_argument("-b", "--bovw-db", required=True,
    help = "Path to where the BOVW DB will be stored")
#ap.add_argument("-d", "--idf", required=True,
#    help = "Path to where the inverse document frequency counts will be stored")
ap.add_argument("-s", "--max-buffer-size", type=int, default=500,
    help = "Max buffer size for # of features to store in memory")
args = vars(ap.parse_args())

# load the codebook vocab and itit the BOVW transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# open the features DB and init the BOVW indexer
featuresDB = h5py.File(args["features_db"], mode="r")
bi = BOVWIndexer(bovw.codebook.shape[0], args["bovw_db"],
    estNumImages=featuresDB["image_ids"].shape[0],
    maxBufferSize=args["max_buffer_size"])

# loop over the image ID's and index
for (i, (imageID, offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):
    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        bi._debug("Processed {} images".format(i), msgType="[PROGRESS]")

    # extract the feature vectors for the current image using the starting and
    # ending offsets (while ignoring keypoints), then quantize the
    # features to construct the BOVW historgram
    features = featuresDB["features"][offset[0]:offset[1]][:, 2:]
    hist = bovw.describe(features)

    # normalize the histogram such that it sums to one then add the
	# bag-of-visual-words to the index
    hist /= hist.sum()
    bi.add(hist)

# close the features DB and finish indexing
featuresDB.close()
bi.finish()

# dump the inverse doc frequency counts to file
#f = open(args["idf"], "wb")
#f.write(pickle.dumps(bi.df(method="idf")))
#f.close()
