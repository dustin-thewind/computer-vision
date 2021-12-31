#import modules
import argparse
import imutils
import cv2

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])
#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\

cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#loop over the contours
for c in cnts:
    #approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)

    #if the approx contour has 4 vertices, then it is a rectangle
    if len(approx) == 4:
        #draw the outline of the countour and draw the text on the image
        cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
        (x, y, w, h) = cv2.boundingRect(approx)
        cv2.putText(image, "rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 2)

cv2.imshow("image", image)
cv2.waitKey(0)
