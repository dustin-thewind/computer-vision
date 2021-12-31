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
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 75, 200)

cv2.imshow("image", image)
cv2.imshow("edge map", edged)

#find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#sort contours from largest to smallest and
#keep only the 7 largest ones
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]

#loop over the contours
for c in cnts:
    #approximate the contour and init the contour colour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.10 * peri, True)

    #show the diff in numbers of vertices, then we have found our rect
    print("original: {}, approx: {}".format(len(c), len(approx)))

    #if the approx contour has 4 vertices, then it is a rectangle
    if len(approx) == 4:
        #draw the outline of the countour on the image
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

#show the output image
cv2.imshow("image", image)
#wait until key press
cv2.waitKey(0)
