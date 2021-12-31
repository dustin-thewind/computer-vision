# USAGE
# python eigenfaces.py --dataset ~/PyImageSearch/Datasets/caltech_faces

# import packages
from __future__ import print_function
from face_recognition.datasets import load_caltech_faces
import resultsmontage
import sklearn
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2

# check sklearn version as the sklearn RandomizedCPA has been deprecated
def is_sklearn_less_than_0_18():
    if int(sklearn.__version__.split(".")[1]) < 18:
        return True
    else:
        return False

# handle if sklearn is < 0.18, where we use RandomizedPCA
if is_sklearn_less_than_0_18():
    print("[INFO] sklearn=={}, so using RandomizedPCA".format(sklearn.__version__))
    from sklearn.decomposition import RandomizedPCA

# else sklearn's RandomizedCPA is deprecated and we need to use PCA
else:
    print("[INFO] sklearn=={}, so using PCA".format(sklearn.__version__))
    from sklearn.decomposition import PCA

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "Path to the faces dataset")
ap.add_argument("-s", "--sample-size", type=int, default=10, help = "# of example samples")
ap.add_argument("-n", "--num-components", type=int, default=32, help = "# principal components")
ap.add_argument("-v", "--visualize", type=int, default=-1, help = "whether or not PCA components should be visualized")
args = vars(ap.parse_args())

# load the caltech data set
print("[INFO] loading faces dataset...")
(training, testing, names) = load_caltech_faces(args["dataset"], min_faces=21, flatten=True,
    test_size=0.25)

# compute the PCA representation of the data, then project the training data
# on to the eigenfaces subspace
print("[INFO] creating eigenfaces...")

# handle if sklearn is < 0.18, where we use RandomizedPCA
if is_sklearn_less_than_0_18():
    pca = RandomizedPCA(n_components=args["num_components"], whiten=True)

# else sklearn is > 0.18
else:
    pca = PCA(svd_solver="randomized", n_components=args["num_components"], whiten=True)

trainData = pca.fit_transform(training.data)

# check to see if the PCA components should be visualized
if args["visualize"] > 0:
    # init the montage for the components
    montage = ResultsMontage((62, 47), 4, 8)

    # loop over the first 16 invidual components
    for (i, component) in enumerate(pca.components_[:8]):
        # reshape the component to a 2d matrix, then convert the data type
        # to an unsigned 8-bit int so it can be displayed with OCV
        component = component.reshape((62,47))
        component = exposure.rescale_intensity(component, out_range=(0, 255)).astype("uint8")
        component = np.dstack([component] * 3)
        montage.addResult(component)

    # show the mean and principal component visualizations
    # show the mean image
    mean = pca.mean_.reshape((62, 47))
    mean = exposure.rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
    cv2.imshow("mean", mean)
    cv2.imshow("components", montage.montage)
    cv2.waitKey(0)

# train a calssifier on the eigenfaces representation
print("[INFO] training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=84)
model.fit(trainData, training.target)

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testing.data))
print(classification_report(testing.target, predictions))

# loop over the desired number of samples
for i in np.random.randint(0, high=len(testing.data), size=(args["sample_size"],)):
    # grab the face and classify it
    face = testing.data[i].reshape((62, 47)).astype("uint8")
    prediction = model.predict(pca.transform(testing.data[i].reshape(1, -1)))

    # resize the face to make it more visible, then display the face and the prediction
    print("[INFO] prediction: {}, actual: {}".format(prediction[0], testing.target[i]))
    face = imutils.resize(face, width=face.shape[1] * 2, inter=cv2.INTER_CUBIC)
    cv2.imshow("face", face)
    cv2.waitKey(0)
