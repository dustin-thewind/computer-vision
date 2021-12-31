#import modules
import argparse
import cv2
import imutils

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

#show the original image
image = cv2.imread(args["image"])
cv2.imshow("orig", image)
cv2.waitKey(0)

#horizontal image flip
flipped = cv2.flip(image, 1)
#cv2.imshow("horizontal flip", flipped)

#flip vertical
flipped2 = cv2.flip(image, 0)
#cv2.imshow("vertical flip", flipped2)

#flip along both axis
flipped3 = cv2.flip(image, -1)
#cv2.imshow("horizontal and vertical flip", flipped3)

#quiz question 1
#horizontal image flip
flipped_h = cv2.flip(image, 1)
cv2.imshow("horizontal flip", flipped_h)

#get the pixel at y = 235, x = 259
(b, g, r) = flipped_h[235, 259]
print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}"
    .format(r=r, g=g, b=b))

cv2.waitKey(0)

#quiz question 2
#horizontal image flip
flipped_h2 = cv2.flip(image, 1)
cv2.imshow("step 1 - q2 h flip", flipped_h2)
#use helper function from imutils
#rotate by 45 counter clockwise
rotated_3 = imutils.rotate(flipped_h2, 45)
cv2.imshow("step 2 - q2 rotate", rotated_3)
#cv2.imshow("rotated -30", rotated)
#horizontal image flip
flipped_v = cv2.flip(rotated_3, 0)
cv2.imshow("step 3 - q2 v flip", flipped_v)
#get the pixel at y = 189, x = 441
(b1, g1, r1) = flipped_v[189, 441]
print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}"
    .format(r=r1, g=g1, b=b1))

cv2.waitKey(0)
