import cv2 as cv
import numpy as np

img = cv.imread('images/shiva.jpg')
# cv.imshow("original", img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([b, blank, r])

b = cv.imshow("blue", blue)
g = cv.imshow("green", green)
r = cv.imshow("red", red)

#merge color chanels
merged = cv.merge((b,g,r))
cv.imshow("merged image", merged)

cv.waitKey(0)