#import modules
import numpy as np
import cv2
import argparse

#create a 300x300 3 channel canvas
#Blue, Green, Red with black background
canvas = np.zeros((300, 300, 3), dtype="uint8")

#draw a green line from tl to br
green = (0, 255, 0)
cv2.line(canvas, (0, 0), (300, 300), green)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)

#draw a 3 pixel thick red line from tr to bl
red = (0, 0, 255)
cv2.line(canvas, (300, 0), (0, 300), red, 3)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)

#draw a 50x50 pixel square, starting at 10x10, ending at 60x60
orange = (0, 165, 255)
cv2.rectangle(canvas, (10, 10), (60, 60), orange)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)

#another rectangle with 5 pixel width
teal = (255, 165, 0)
cv2.rectangle(canvas, (50, 200), (200, 225), teal, 5)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)

#yet another rectangle, this time filled in with violet
violet = (211, 0, 148)
cv2.rectangle(canvas, (200, 50), (225, 125), violet, -1)
cv2.imshow("canvas", canvas)
cv2.waitKey(0)

#create a new canvas and draw some circles
#of increasing radii - 25px to 150px
canvas2 = np.zeros((300, 300, 3), dtype="uint8")
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)

for r in range(0, 175, 25):
    cv2.circle(canvas2, (centerX, centerY), r, white)

cv2.imshow("canvas2", canvas2)
cv2.waitKey(0)

#create a new canvas and draw some abstract circles
#of increasing radii - 25px to 150px
canvas3 = np.zeros((300, 300, 3), dtype="uint8")

for i in range(0, 25):
    #randomly generate a radius between 5 and 200
    #randomly generate a color
    #pick a random spot on the canvas to draw the circle
    radius = np.random.randint(5, high = 200)
    color = np.random.randint(5, high = 256, size = (3,)).tolist()
    pt = np.random.randint(0, high=300, size = (2,))
    #draw the random circle
    cv2.circle(canvas3, tuple(pt), radius, color, -1)

#display out the random circle
cv2.imshow("canvas3", canvas3)
cv2.waitKey(0)

#set up argurment parser for image to draw on
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

#show the original image
image = cv2.imread(args["image"])
cv2.imshow("original", image)
cv2.waitKey(0)

cv2.circle(image,(168, 188), 90, (0, 0, 255), 2)
cv2.circle(image,(150, 164), 10, (0, 0, 255), -1)
cv2.circle(image,(192, 174), 10, (0, 0, 255), -1)
cv2.rectangle(image, (134, 200), (186, 218), (0, 0, 255), -1)

cv2.imshow("modified", image)
cv2.waitKey(0)

#create a 300w X 100h 3 channel canvas
#Blue, Green, Red with black background
canvas_test = np.zeros((100, 300, 3), dtype="uint8")
cv2.imshow("canvas test", canvas_test)
cv2.waitKey(0)
