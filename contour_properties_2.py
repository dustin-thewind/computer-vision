#import modules
import argparse
import imutils
import numpy as np
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

thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]

#show original image
cv2.imshow("orig", image)
cv2.imshow("thresh", thresh)

#find external contours in the thresh'd image and allocate memory
#for the convex hull image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
hullImage = np.zeros(gray.shape[:2], dtype="uint8")

#loop over the contours
for (i, c) in enumerate(cnts):
    #compute the area of the countour along with the bounding box
    #to compute the aspect ratio
    area = cv2.contourArea(c)
    (x, y, w, h) = cv2.boundingRect(c)

    #compute the aspect ratio of the countour
    # which is width divided by height of bounding box
    aspectRatio = w / float(h)
    print("aspect ratio{}".format(aspectRatio))

    #use the area of the contour and the bounding box to
    #compute the extent
    extent = area / float(w * h)

    #compute the convexHull of the contour, then use the area of the
    #original contour and the area of the convexHull to compute the
    #solidity
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)

    #visualize the original contours and the convex hull
    #and initialize the name of the shape
    cv2.drawContours(hullImage, [hull], -1, 255, -1)
    cv2.drawContours(image, [c], -1, (240, 0, 159), 3)
    shape = ""

    #if the aspect ration is approx 1, shape is a square
    if aspectRatio >= 0.98 and aspectRatio <= 1.02:
        shape = "SQUARE"

    #if width is 3x the height, then it is a rectangle
    elif aspectRatio >= 3.0:
        shape = "RECT"

    #if extent is small, we have an l-piece
    elif extent < 0.65:
        shape = "L piece"

    #if the solidity is sufficiently high, we have a z-piece
    elif solidity > 0.80:
        shape = "Z piece"

    #draw the shape name on the image
    cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (240, 0, 159), 2)

    #show the contour properties
    print("Countour #{} -- aspect_ratio={:.2f}, extent={:.2f}, solidity={:.2f}"
        .format(i + 1, aspectRatio, extent, solidity))

    #show the output
    cv2.imshow("convex hull", hullImage)
    cv2.imshow("img", image)
    cv2.waitKey(0)
