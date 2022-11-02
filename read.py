import cv2 as cv

img = cv.imread('images/space.jpg')

cv.imshow('space', img)

cv.waitKey(0)