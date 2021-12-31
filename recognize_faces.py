# import the necessary packages
from face_recognition import FaceRecognizer
from face_recognition import FaceDetector
import argparse
import imutils
import cv2

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face-cascade", required=True, help = "path to the face detection cascade")
ap.add_argument("-c", "--classifier", required=True, help = "path to the classifier")
ap.add_argument("-t", "--confidence", type=float, default=100.0,
    help = "max confidence threshold for positive face identification")
args = vars(ap.parse_args())

# init the face detector, load the face recognizer, and set the confidence
# threshold
fd = FaceDetector(args["face_cascade"])
fr = FaceRecognizer.load(args["classifier"])
fr.setConfidenceThreshold(args["confidence"])

# grab a reference to the webcam
# grab a reference to the webcam and open the output file for writing
camera = cv2.VideoCapture(0)

# loop over the video frames
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if the frame could not be grabbed, then we've reached the end of the video
    if not grabbed:
        break

    # resize the frame, convert the fram to grayscale, and detect faces in the frame
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # loop over the face bounding boxes
    for (i, (x, y, w, h)) in enumerate(faceRects):
        # grab the face to predict
        face = gray[y:y + h, x:x + w]

        # predict who's face it is, display the text on the image,
        # and draw a bounding box around the face
        (prediction, confidence) = fr.predict(face)
        prediction = "{}: {:.2f}".format(prediction, confidence)
        cv2.putText(frame, prediction, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the frame and record if the user presses a key
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, break the loop
    if key == ord("q"):
        break

# clean up and close any open windows
camera.release()
cv2.destroyAllWindows()
