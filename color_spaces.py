
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

#load the image and display RGB values
#visualize as a cube
for (name, chan) in zip(("B", "G", "R"), cv2.split(image)):
    cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)

#load the image and display HSV values
#visualize as a cylinder
#hue - what the 'pure' color is
#saturation - how 'white' the color is
#a fully saturated color is 'pure', as in 'pure' blue
#a color with 0 saturation would be pure white
#value - the lightness of the color
#value of 0 indicates pure black, higher value produces lighter colors
for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv)):
    cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)

#load the image and display L*a*b* values
#to visualize think of a sphere
#L-channel - the 'lightness' of the pixel
#this travels up and down the vertical axis
#a-channel - orignates from the center of the L-channel
#and defines a pure green on one end of the spectrum
#and a pure red on the other end
#b-channel - orignates from the center of the L-channel
#and runs perpendicular to the a-channel
#defines a pure blue at one end of the spectrum
#and pure yellow at the other end
for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(lab)):
    cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()

#look at the grayscale version of the image
#recall that grayscale IS NOT black and white
#as black and white are binary and can only have 2 values, 0 or 255
#grayscale images are single channel images with pixel values from 0 to 255
#we apply weighting to the conversion from RGB to grayscale
# Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
#grayscale is used when we don't care about color
#such as detecting faces or building object classifiers
#this allows us to save space and is more computationally efficient
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey(0)

gray = cv2.cvtColor((156, 107, 81), cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
print(gray)
cv2.waitKey(0)
