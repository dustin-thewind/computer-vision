#import packages
import cv2
import numpy as np

#draw a rectangle
rectangle = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("rectangle", rectangle)

#draw a circle
circle = np.zeros((300, 300), dtype="uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("circle", circle)

#cv2.waitKey(0)

#bitwise AND is only TRUE when both rect and circle
#have a value that is 'ON'
#bitwise_and examines every pixel in the rect and circle
#if both px have value > 0 then that pixel is turned 'ON'
# if a pixel is ON its value is set to 255 in the output image
#if both px are not >0 then the output pixel is 'OFF', with a value of 0
bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("and", bitwiseAnd)
cv2.waitKey(0)

#bitwise 'XOR' examines each pixel in rect and cirlce
#if either pixel in circ or rect > 0 then
#the output pixel value is set to 255, otherwise it is 0
bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey(0)

#bitwise 'NOT' inverts the values of both px
#px with a value of 255 become 0
#and px with a value of 0 become 255
bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("not", bitwiseNot)
cv2.waitKey(0)
