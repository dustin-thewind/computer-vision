#import modules
from matplotlib import pyplot as plt
import argparse
import cv2
import numpy as np

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])
#convert to grayscale

def plot_histogram(image, title, mask=None):
    #grab the image channels, init the tuple
    #of colors and the figure
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("# of pix")

    #loop over the image channels
    for (chan, color) in zip(chans, colors):
        #create hist for the current chan and plot
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

cv2.imshow("orig", image)
plot_histogram(image, "hist for original")

#build a mask for the image
#black for regions to ignore, white for regions to examine
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (60, 290), (210, 390), 255, -1)
cv2.imshow("Mask", mask)

#show the masked image
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("masked", masked)

plot_histogram(image, "hist for masked img", mask=mask)

#show the plots
plt.show()

cv2.waitKey(0)
