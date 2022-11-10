'''
thresholding -> binarization of images
'''
import cv2 as cv

img = cv.imread('images/shiva.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow("threshold image", thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow("inversed thresh image", thresh_inv)

#adeptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 5)
cv.imshow("adaptive threshold img", adaptive_thresh)


adaptive_thresh_inv = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 13, 5)
cv.imshow("adaptive threshold img inverse", adaptive_thresh_inv)
cv.waitKey(0)