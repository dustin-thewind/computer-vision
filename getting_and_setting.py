#import modules
import argparse
import cv2

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

#show the original image
image = cv2.imread(args["image"])

#slice the array to find the height, width
#recall the array is a tuple of h, w, and # channels
(h, w) = image.shape[:2]
cv2.imshow("Original", image)
#cv2.waitKey(0)

#print top left pixel in the NumPy array
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}"
    .format(r=r, g=g, b=b))

#change the value of the pixel at [0, 0]
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}"
    .format(r=r, g=g, b=b))

#computer the center of the image
#width and height divided by two
(cX, cY) = (w // 2, h // 2)

#slice the array to get top left Corner
#start slice at on y axis startY, stop at ending y value
#startY:endY
#similar for X axis, startY:endY
#slice sections of the image
tl = image[0:cY, 0:cX]
tr = image[0:cY, cX:w]
br = image[cY:h, cX:w]
bl = image[cY:h, 0:cX]
#display all the slices
cv2.imshow("top left", tl)
cv2.imshow("top right", tr)
cv2.imshow("bottom right", br)
cv2.imshow("bottom left", bl)
cv2.waitKey(0)

image[0:cY, 0:cX] = (0, 255, 0)
cv2.imshow("updated", image)
cv2.waitKey(0)
