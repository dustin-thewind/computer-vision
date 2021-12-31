# USAGE
# python cluster_histograms.py --dataset dataset

# import packages
from lab_10_3.descriptors.labhistogram import LabHistogram
from sklearn.cluster import KMeans
from imutils import paths
import numpy as np
import argparse
import cv2

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
    help = "Path to input data directory")
ap.add_argument("-k", "--clusters", type=int, default=2,
    help = "# of clusters to generate")
args = vars(ap.parse_args())

# init the image descriptor along with the img matrix
desc = LabHistogram([8, 8, 8])
data = []

# grab the image paths fromt he dataset dir
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.array(sorted(imagePaths))

for imagePath in imagePaths:
    # load the image, describe it, the update the list
    image = cv2.imread(imagePath)
    hist = desc.describe(image)
    data.append(hist)

# cluster the color hists
clt = KMeans(n_clusters=args["clusters"])
labels = clt.fit_predict(data)

# loop over the unique labels
for label in np.unique(labels):
    # grab all img paths that are assigned to the current label
    labelPaths = imagePaths[np.where(labels == label)]

    # loop over the image paths that belong to the current label
    for (i, path) in enumerate(labelPaths):
        # load and display the image
        image = cv2.imread(path)
        cv2.imshow("Cluster {}, Image #{}".format(label + 1, i + 1), image)

    # wait for keypress and then close all open windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
