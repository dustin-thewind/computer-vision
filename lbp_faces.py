# USAGE
# python lbp_faces.py --dataset caltech_faces

# import packages
from __future__ import print_function
from face_recognition.datasets import load_caltech_faces
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import argparse
import imutils
import cv2

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "Path to the faces dataset")
ap.add_argument("-s", "--sample-size", type=int, default=10, help = "# of example samples")
args = vars(ap.parse_args())

print("[INFO] loading faces dataset...")
(training, testing, names) = load_caltech_faces(args["dataset"], min_faces=21,
    test_size=0.25)

# encode the labels, transforming them from strings to integers
# - OCV does not like strings as training data
le = LabelEncoder()
le.fit_transform(training.target)

# handle if we are creating the LBP face recognizer for OCV 2.4
if imutils.is_cv2():
    recognizer = cv2.createLBPHFaceRecognizer(radius=2, neighbors=16, grid_x=8, grid_y=8)

# otherwise we are creating the LBP face recognizer for OCV 3+
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=3, grid_y=3)

# train the LBP face recognizer
print("[INFO] training face recognizer...")
recognizer.train(training.data, le.transform(training.target))

# init the list of prediction and confidence scores
print("[INFO] gathering predictions...")
predictions = []
confidence = []

# loop over the test data
for i in range(0, len(testing.data)):
    print("{} of {}".format(str(i), str(len(testing.data))))
    # classify the face and update the list of predictions and confidence scores
    (prediction, conf) = recognizer.predict(testing.data[i])
    predictions.append(prediction)
    confidence.append(conf)

# show the classifiction report
print(classification_report(le.transform(testing.target), predictions,
    target_names=np.unique(names)))

# loop over the desired number of samples
for i in np.random.randint(0, high=len(testing.data), size=(args["sample_size"],)):
    # resize the face to make it more visible, then display
    # the face and the prediction
    print("[INFO] Prediction: {}, Actual: {}, Confidence: {:.2f}".format(
        le.inverse_transform([predictions[i]])[0], testing.target[i], confidence[i]))
    face = testing.data[i]
    face = imutils.resize(face, width=face.shape[1] * 2, inter=cv2.INTER_CUBIC)
    cv2.imshow("face", face)
    cv2.waitKey(0)
