# USAGE
# python detect_faces_video.py -f /cascades/haarcascade_frontalface_default.xml

# import the necessary packages
import argparse
import cv2
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="Path to where the face cascade resides")
ap.add_argument("-v", "--video", help="Path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector and detect faces in the image
detector = cv2.CascadeClassifier(args["face"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and did not grab a
    # frame then we have reached the end of thev video
    if args.get("video") and not grabbed:
        break

    # resize the frame, convert it to grayscale
    # and detect faces in the grame
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # handle face detection for OpenCV 2.4
    if imutils.is_cv2():
    	faceRects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
    		minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    # otherwise handle face detection for OpenCV 3+
    else:
    	faceRects = detector.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=5,
    		minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the faces and draw a rectangle around each
    for (x, y, w, h) in faceRects:
    	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the frame on our screen
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# clean up the camera and close any open windows
camera.release()
cv2.destroyAllWindows(0)
