#import modules
import argparse
import cv2
import numpy as np
import imutils

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])
#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#show original image
cv2.imshow("orig", image)

cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
clone = image.copy()

#loop of the contours
for c in cnts:
    #compute the moments of the countour which
    #can be used to compute the centroid of the region
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print("centroid at x{}, y{}".format(cX, cY))
    #draw the center of the countour on the image
    cv2.circle(clone, (cX, cY), 10, (0, 255, 0), -1)

#show the output
cv2.imshow("centroids", clone)
cv2.waitKey(0)

#clone the orignal image
clone = image.copy()

#loop over the contours again
for (i, c) in enumerate(cnts):
    #compute the area and perimeter of each contour
    area = cv2.contourArea(c)
    #arcLength takes two arguments, the countour itself
    #and a flag that indicates if the contour is 'closed'
    #a contour is considered closed if the shape is continuous
    #and does not have any gaps
    perimeter = cv2.arcLength(c, True)
    print("countour #{} -- area: {:.2f}, perimeter: {:.2f}".format(i + 1, area, perimeter))
    #draw the countour on the image
    cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
    #compute the center of the contour and draw the contour number
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(clone, "#{}".format(i + 1), (cX -20, cY), cv2.FONT_HERSHEY_SIMPLEX,
    1.25, (255, 255, 255), 4)

cv2.imshow("contours", clone)
cv2.waitKey(0)

#clone the orignal image
clone = image.copy()

#loop over the contours
for c in cnts:
    #fit a bounding box to the contour
    (x, y, w, h) = cv2.boundingRect(c)
    print("box dimensions x{}, y{}, w{}, h{}".format(x, y, w, h))
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("bounding box", clone)
cv2.waitKey(0)

clone = image.copy()

#loop over the contours
for c in cnts:
    #fit a rotated bounding box to the contour and draw it
    #minAreaRect returns a tuple with 3 values
    #first value is starting x,y coordinate of the rotated bounding box
    #second is height and width of the box
    #third is the angle of the rotation of the shape
    box = cv2.minAreaRect(c)
    box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
    cv2.drawContours(clone, [box], -1, (0, 255, 0), 2)

cv2.imshow("rotated bounding box", clone)
cv2.waitKey(0)

clone = image.copy()

for c in cnts:
    #fit a minimum enclosing circle to the contour
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    print("radius of min circle r{}".format(radius))
    cv2.circle(clone, (int(x), int(y)), int(radius), (0, 255, 0), 2)

#show output
cv2.imshow("min-enclosing circles", clone)
cv2.waitKey(0)

clone = image.copy()

for c in cnts:
    #to fit an ellipse, the contour must have at least 5 points
    if len(c) >=5:
        ellipse = cv2.fitEllipse(c)
        cv2.ellipse(clone, ellipse, (0, 255, 0), 2)


#show output
cv2.imshow("ellipses", clone)
cv2.waitKey(0)
