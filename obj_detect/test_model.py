# USAGE
# python test_modeli.py -c conf/cars.json images/image.jpe

# import packages
from object_detection import ObjectDetector
from object_detection import non_max_suppression
from descriptors import HOG
from utils import Conf
import numpy as np
import imutils
import argparse
import pickle
import cv2

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="Path config file")
ap.add_argument("-i", "--image", required=True, help="Path to the image to be classified")
args = vars(ap.parse_args())

# load the config file
conf = Conf(args["conf"])

# load the classifier, then init the HOG descriptor
# and object detector
model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"], block_norm="L1")
od = ObjectDetector(model, hog)

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(260, image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect objects in the image
(boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["window_step"],
    pyramidScale=conf["pyramid_scale"]
    , minProb=conf["min_probability"])
pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
orig = image.copy()

print("# of bounding boxes: {}".format(len(boxes)))

# loop over the bounding boxes and draw them
for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

for (startX, startY, endX, endY) in pick:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output images
cv2.imshow("image:{}".format(args["image"]), image)
cv2.imshow("orig", orig)
cv2.waitKey(0)
