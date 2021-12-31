# USAGE
# python test_model_classifier.py -i test_images -c output/vocab.cpickle -m output/model.cpickle

# import packages
from __future__ import print_function
from sklearn.metrics import classification_report
from prod_cbir.descriptors import DetectAndDescribe
from prod_cbir.descriptors import PBOW
from prod_cbir.ir import BagOfVisualWords
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import numpy as np 
import argparse
import pickle
import imutils
import cv2

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the images directory")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-l", "--num-levels", type=int, default=2, help="# of pyramid levels to generate")
ap.add_argument("-m", "--model", required=True, help="Path the classifier")
args = vars(ap.parse_args())

# init the keypoint detector, local invariant descriptor
# and the descriptor pipeline
detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

# load the codebook vocab and init the BOVW transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)
pbow = PBOW(bovw, numLevels=args["num_levels"])

# load the classifier and grab the list of image paths
model = pickle.loads(open(args["model"], "rb").read())
imagePaths = list(paths.list_images(args["images"]))

# init the list of true labels and predicted labels
print("[INFO] extracting features from testing data...")
trueLabels = []
predictedLabels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the true label from the image path and update the true labels list
    trueLabels.append(imagePath.split("/")[-2])

    # load the image and prepare it from description
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=min(320, image.shape[1]))

    # describe the image and classify it
    (kps, descs) = dad.describe(gray)
    hist = pbow.describe(gray.shape[1], gray.shape[0], kps, descs)
    prediction = model.predict(hist)[0]
    predictedLabels.append(prediction)

# show the classification report
print(classification_report(trueLabels, predictedLabels))

# loop over a sample of the testing images
for i in np.random.choice(np.arange(0, len(imagePaths)), size=(20,), replace=False):
    # load the image and show the prediction
    image = cv2.imread(imagePaths[i])

    # show the prediction
    filename = imagePaths[i][imagePaths[i].rfind("/") + 1:]
    print("[PREDICTION] {}: {}".format(filename, predictedLabels[i]))
    cv2.putText(image, predictedLabels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (0, 255, 0), 2)
    cv2.imshow("Image {}".format(filename), image)
    cv2.waitKey(0)
