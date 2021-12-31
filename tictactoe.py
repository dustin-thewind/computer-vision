#import modules
import argparse
import cv2
import imutils

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", "-img", required = True, help = "Path to the tictactoe image file")

args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])
#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#loop over the contours
for (i, c) in enumerate(cnts):
    #compute the area of the contour along with the bounding box
    #to compute the aspect ratio
    area = cv2.contourArea(c)
    (x, y, w, h) = cv2.boundingRect(c)

    #compute the convex hull of the contour, then use the area of the
    #original contour and the area of the convex hull to compute the solidity
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)

    #initialize char
    char = "?"

    #if the solidity is high, then we are examinining an 'o'
    if solidity > 0.9:
        char = "o"
    #if the solidity is reasonably high, we are examining an 'x'
    elif solidity > 0.5:
        char = "x"
    #if the character is not unknown, draw it
    if char != "?":
        cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
        cv2.putText(image, char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
        (0, 255, 0), 4)

    #print the contour properties
    print("{} (Contour#{}) -- solidity={:.2f}".format(char, i + 1, solidity))

#show output
cv2.imshow("Output", image)
cv2.waitKey(0)
