import cv2 as cv
import numpy as np

#read images
img = cv.imread('images/shiva.jpg')
cv.imshow('shiva', img)
'''
#converting to grayscale
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('grayscale image', gray_img)

#blur images
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow("blur image", blur)

#edge cascade
canny = cv.Canny(img, 125, 175)
cv.imshow("canny", canny)

#dilating images
dilated = cv.dilate(canny, (3,3), iterations=2)
cv.imshow("dilated image", dilated) 

#eroding image
eroded = cv.erode(dilated, (3,3), iterations=3)
cv.imshow('eroded image', eroded)
'''

#resizing image
resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
cv.imshow("resized image", resized)

#cropping 
cropped = img[50:200, 200:400] 
cv.imshow('cropped images', cropped)

cv.waitKey(0)