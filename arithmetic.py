#import modules
import argparse
import cv2
import numpy as np

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("orig", image)
cv2.waitKey(0)

#images are numpy arrays, stored as 8bit unsigned ints
#values for pixels range within [0, 255]
#cv2.add and cv2.subtract clips values within the range
#example: 250 + 10 would be clipped to 255, 5 - 10 would be clipped to 0
print("max of 255: {}".format(str(cv2.add(np.uint8([200]), np.uint8([100])))))
print("min of 0: {}".format((str(cv2.subtract(np.uint8([50]), np.uint8([100]))))))

#using numpy arithmetic on these arays will be modulo (wrap around)
#instead of being clipped
print("wrap around: {}".format(str(np.uint8([228]) + np.uint8([-78]))))
print("wrap around: {}".format(str(np.uint8([1]) - np.uint8([251]))))

#increase the pixel intensity of all pixels by 100
#construct a numpy array that is the same size of our matrix
#fill with 1's, multiple by 100 to fill with 100's
#add the images together
M = np.ones(image.shape, dtype="uint8") * 100
added = cv2.add(image, M)
cv2.imshow("Added", added)

#decrease pixel intensity by 50
#construct a numpy array that is the same size of our matrix
#fill with 1's, multiple by 100 to fill with 100's
#add the images together
M = np.ones(image.shape, dtype="uint8") * 50
subtracted = cv2.subtract(image, M)
cv2.imshow("subtracted", subtracted)

#quiz question 5
#increase the pixel intensity of all pixels by 75
#construct a numpy array that is the same size of our matrix
#fill with 1's, multiple by 75 to fill with 75's
#add the images together
M = np.ones(image.shape, dtype="uint8") * 75
added = cv2.add(image, M)
cv2.imshow("Added", added)

#get the pixel at y = 152, x = 61
(b, g, r) = added[152, 61]
print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}"
    .format(r=r, g=g, b=b))

cv2.waitKey(0)
