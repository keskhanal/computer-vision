import cv2 as cv
import numpy as np

img = cv.imread('images/shiva.jpg')
cv.imshow("original image", img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow("blank image", blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale image", gray)

blur = cv.GaussianBlur(gray,(5,5), cv.BORDER_DEFAULT)
cv.imshow("blur image", blur)

canny = cv.Canny(blur, 150, 150)
cv.imshow("Canny edges", canny)

# ret, thres = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
# cv.imshow("thres image", thres)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f"{len(contours)} contours found!!!")

cv.drawContours(blank, contours, -1, (0,0,255), 2)
cv.imshow("contours drawn", blank)

cv.waitKey(0)