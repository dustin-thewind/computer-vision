#import modules
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

#set up the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())
# Load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

#build histogram of the original image
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("Original Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# Pixels")

#Loop over the image channels to build the histogram
#for the original image
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0,256])

print("The contents of our 2d array:\n",image)
#wait until key press
cv2.waitKey(0)

#image is an NumPy array of height, width and # channels
#each item in the array is a tuple
#of height(Y), width(X), and
(h, w, c) = image.shape
print("Array Dimensions and Channels: X(Height):{h}, Y(width):{w}, Channels:{c}"
    .format(h=h, w=w, c=c))
cv2.waitKey(0)

#how many pixels are in our image
print("# pixels: {}".format(str(image.size)))
cv2.waitKey(0)

#print RGB value of px at y = 100, x = 75
(b, g, r) = image[100, 75]
print("Pixel at (100, 75) - Red: {r}, Green: {g}, Blue: {b}"
    .format(r=r, g=g, b=b))
cv2.waitKey(0)

#the value of the red pixel at y = 150, x = 120
#before modifying the image
print("red pixel at y = 150, x = 120: {}".format(image.item(150, 120, 2)))
cv2.waitKey(0)

#increase the pixel intensity of all pixels
#construct a numpy array that is the same size of our original image
#fill with the new array with 1's, multiply by 50 to fill with 50's
#add the arrays
matrix = np.ones(image.shape, dtype="uint8") * 50
increased = cv2.add(image, matrix)
#show the modified (increased) image
cv2.imshow("Increased/Added", increased)
#what is the value of the red pixel at y = 150, x = 120
#after increasing intensity
print("INCREASED by 50 red pixel at y = 150, x = 120: {}".format(increased.item(150, 120, 2)))
cv2.waitKey(0)

#build the histogram of the increased image
chans = cv2.split(increased)
colors = ("b", "g", "r")
plt.figure()
plt.title("Increased/Added Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# Pixels")

# Loop over the image channels
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0,256])

#decrease pixel intensity
#construct a numpy array that is the same size of our matrix
#fill with 1's, multiple by 50 to fill with 50's
#subtract new array from original array
matrix = np.ones(image.shape, dtype="uint8") * 50
decreased = cv2.subtract(image, matrix)
#show the modified (decreased) image
cv2.imshow("Subtracted/Decreased", decreased)
print("DECREASED by 50 red pixel at y = 150, x = 120: {}".format(decreased.item(150, 120, 2)))
cv2.waitKey(0)

#build the histogram of the decreased image
chans = cv2.split(decreased)
colors = ("b", "g", "r")
plt.figure()
plt.title("Subtracted/Decreased Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# Pixels")

# Loop over the image channels
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0,256])

#add the two modified images together
merged = cv2.add(increased, decreased)
cv2.imshow("Merged by Addition", merged)
print("MERGED(Added) red pixel at y = 150, x = 120: {}".format(merged.item(150, 120, 2)))
cv2.waitKey(0)

#build the histogram of the merged image
chans = cv2.split(merged)
colors = ("b", "g", "r")
plt.figure()
plt.title("Merged by Addition Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# Pixels")

# Loop over the image channels
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0,256])

#show all the histogram plots we built
plt.show()

#wait for key press before exit
cv2.waitKey(0)
