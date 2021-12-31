# import packages
from __future__ import print_function
from prod_cbir.ir import BagOfVisualWords
from sklearn.metrics import pairwise
import numpy as np

# randomly generate the Vocabulary/cluster centers along with
# the feature vectors -- generate 10 feature vectors containing
# 6 real-valued entries, along with a codebook containing 3
# visual words
np.random.seed(42)
vocab = np.random.uniform(size=(3, 6))
features = np.random.uniform(size=(10, 6))
print("[INFO] vocabulary:\n{}\n".format(vocab))
print("[INFO] features:\n{}\n".format(features))

# init our bag of visual words histogram with 3 entries
# one for each of the possible visual words
hist = np.zeros((3,), dtype="int32")

# loop over the individual feature vectors
for (i, f) in enumerate(features):
    # compute the euclidean distance between the current feature vector
    # and the 3 visual words; then, find the index of the visual word
    # with the smallest distance
    D = pairwise.euclidean_distances(f.reshape(1, -1), Y=vocab)
    j = np.argmin(D)

    print("[INFO] closest visual word to feature #{}: {}".format(i, j))
    hist[j] += 1
    print("[INFO] updated histogram: {}".format(hist))

# apply the BagOfVisualWords class to speed up the process
bovw = BagOfVisualWords(vocab, sparse=False)
hist = bovw.describe(features)
print("[INFO] BOVW histogram: {}".format(hist))
