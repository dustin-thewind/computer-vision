# USAGE
# python train_model_classifier.py -d caltech5 -f output/features_caltech5.hdf5 /
#   -b output/bovw_caltech5.hdf5 -m output/model.cpickle

# import packages
from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import sklearn
import numpy as np
import argparse
import pickle
import h5py
import cv2

# handle sklearn versions <0.18
if int((sklearn.__version__).split(".")[1]) < 18:
    from sklearn.grid_search import GridSearchCV

# otherwise sklearn.grid_search is deprecated
# so import GridSearchCV from sklearn.model_selection
else:
    from sklearn.model_selection import GridSearchCV

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help = "Path to directory that contains the original images")
ap.add_argument("-f", "--features-db", required=True,
    help = "Path to the features DB")
#ap.add_argument("-b", "--bovw-db", required=True,
#    help = "Path to the BOVW DB")
ap.add_argument("-p", "--pbow-db", required=True,
    help = "Path to the PBOW DB")
ap.add_argument("-m", "--model", required=True,
    help = "Path to the output classifier")
args = vars(ap.parse_args())

# open the features and bovw DB's
featuresDB = h5py.File(args["features_db"])
bovwDB = h5py.File(args["pbow_db"])

# grab the training and testing data from the dataset using the
# first 300 images as training data, and the remaining 200 for testing
print("[INFO] loading data...")
(trainData, trainLabels) = (bovwDB["bovw"][:300], featuresDB["image_ids"][:300])
(testData, testLabels) = (bovwDB["bovw"][300:], featuresDB["image_ids"][300:])

# prepare the labels by removing the filename from the image ID
# leaving just the class name
trainLabels = [l.split(":")[0] for l in trainLabels]
testLabels = [l.split(":")[0] for l in testLabels]

# define the grid of parameters to explore, then start the grid search where
# we evaluate the linear SVM for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LinearSVC(random_state=42), params, cv=3)
model.fit(trainData, trainLabels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# show the classification report
print("[INFO] evaluating...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

# loop over a sample of the testing data
for i in np.random.choice(np.arange(300, 375), size=(20,), replace=False):
    # randomly grab a testing image, load it, and classify it
    (label, filename) = featuresDB["image_ids"][i].split(":")
    image = cv2.imread("{}/{}/{}".format(args["dataset"], label, filename))
    prediction = model.predict(bovwDB["bovw"][i].reshape(1, -1))[0]

    # show the prediction
    print("[PREDICTION] {}: {}".format(filename, prediction))
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("image {}".format(filename), image)
    cv2.waitKey(0)

# close the DB's
featuresDB.close()
bovwDB.close()

# dump the classifier to file
print("[INFO] dumping classifier to file...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()
