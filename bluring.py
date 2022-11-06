import cv2 as cv

#read image
img = cv.imread('images/shiva.jpg')
cv.imshow("original", img)

#average bluring
avg_blur = cv.blur(img, (7,7))
cv.imshow("average blur", avg_blur)

#gaussian blur
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow("gaussian blur", gauss)

#median blur
median_blur = cv.medianBlur(img, 3)
cv.imshow("median blur", median_blur)

#bilateral blur
bilat_blur = cv.bilateralFilter(img, 10, 15, 15)
cv.imshow("bilateral blur", bilat_blur)

cv.waitKey(0)