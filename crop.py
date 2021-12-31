#import modules
import argparse
import cv2

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("orig", image)
cv2.waitKey(0)

#slice the numpy array
#startY:endY , startX:endX
face = image[85:250, 85:220]
cv2.imshow("Face", face)
cv2.waitKey(0)

body = image[90:450, 0:290]
cv2.imshow("Body", body)
cv2.waitKey(0)

#quiz question 2
#slice the numpy array
#startY:endY , startX:endX
people = image[124:212, 225:380]
cv2.imshow("people", people)
cv2.waitKey(0)
