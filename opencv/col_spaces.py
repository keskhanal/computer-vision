import cv2 as cv

img = cv.imread('images/shiva.jpg')
cv.imshow("original img", img)

#bgr to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("grayscale image", gray)

#bgr to hsv
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("hsv", hsv)

#bgr to lab
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow("lab img", lab)

#bgr to rgb
bgr_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("bgr_rgb image", bgr_rgb)

cv.waitKey(0)