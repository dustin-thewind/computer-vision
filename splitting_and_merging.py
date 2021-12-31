
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

(B, G, R) = cv2.split(image)

#show each channel individually
#this will show a channel representation of the image
#which will appear to be greyscale
cv2.imshow("red", R)
cv2.imshow("green", G)
cv2.imshow("blue", B)
cv2.waitKey(0)

#merge the split images back together
merged = cv2.merge([B, G, R])
cv2.imshow("merged", merged)
cv2.waitKey(0)

#visualize each channel with color
#slice the image to get the dimensions
#set zeros to all channels we are not interested
#in visualizing
zeros = np.zeros(image.shape[:2], dtype="uint8")
cv2.imshow("red", cv2.merge([zeros, zeros, R]))
cv2.imshow("green", cv2.merge([zeros, G, zeros]))
cv2.imshow("blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)

#quiz question 1
#the value of the red pixel at y = 94, x = 180
#before modifying the image
print("red pixel at y = 94, x = 180: {}".format(image.item(94, 180, 2)))
cv2.waitKey(0)

#quiz question 2
#the value of the blue pixel at y = 78, x = 13
#before modifying the image
print("blue pixel at y = 78, x = 13: {}".format(image.item(78, 13, 0)))
cv2.waitKey(0)

#quiz question 3
#the value of the green pixel at y = 5, x = 80
#before modifying the image
print("red pixel at y = 5, x = 80: {}".format(image.item(5, 80, 1)))
cv2.waitKey(0)
