# USAGE
# python test_detector.py --detector output/stop_sign_detector.svm --testing stop_sign_testing

# import the necessary packages
import argparse
import cv2
from imutils import paths
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="Path to the trained object detector")
ap.add_argument("-t", "--testing", required=True, help="Path to the directory of testing images")
args = vars(ap.parse_args())

# load the detector
detector = dlib.simple_object_detector(args["detector"])

for testingPath in paths.list_images(args["testing"]):
    # load the image and make predictions
    image = cv2.imread(testingPath)
    boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # loop over the bounding boxes and draw them
    for b in boxes:
        (x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
        print("Bounding box coordinates: Left:{}, Top:{}, Right:{}, Bottom:{}".format(x, y, w, h))
        cv2.rectangle(image, (x, y), (w, h), (12, 255, 103), 2)

    # show the image
    cv2.imshow("image", image)
    cv2.waitKey(0)
