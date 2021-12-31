#import modules
from matplotlib import pyplot as plt
import argparse
import cv2
import imutils

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])

#grab the image channels, init the tuple of colors and the figure
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("Flattened color hist")
plt.xlabel("# of bins")
plt.ylabel("# of pix")

#loop over the image channels
for (chan, color) in zip(chans, colors):
    #create a histogram for the current channel and plot
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

#reduce the number of bins in the hist from 256 to 32
#to better visualize the results
fig = plt.figure()

#plot a 2d color hist for green and blue
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32],
    [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D hist for G & B")
plt.colorbar(p)

#plot a 2d color hist for green and blue
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32],
    [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D hist for G & R")
plt.colorbar(p)

#plot a 2d color hist for blue and red
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32],
    [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D hist for B & R")
plt.colorbar(p)

#examine dimensionality of one of the 2d histograms
print("2d hist shape: {}, with {} values".format(hist.shape,
    hist.flatten().shape[0]))

#build a 3d color hist with 8 bins in each direction
#we can plot a 3d histogram, so we'll just show the shape of the hist
hist = cv2.calcHist([image], [0, 1, 2], None,
    [9, 16, 8], [0, 256, 0, 256, 0, 256])
print("3d hist shape: {}, with {} values".format(hist.shape,
    hist.flatten().shape[0]))

#display the image with matplotlib
plt.figure()
plt.axis("off")
plt.imshow(imutils.opencv2matplotlib(image))

#show the plots
plt.show()
