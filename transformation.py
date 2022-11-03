import cv2 as cv
import numpy as np

img = cv.imread('images/shiva.jpg')

cv.imshow("shiva image", img)

#translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimensions = (img.shape[1], img.shape[0])

    return cv.warpAffine(img, transMat, dimensions)

# -x --> left
# -y --> up
# x --> right
# y --> down

translated = translate(img, -100, 100)
cv.imshow('translated img', translated)


#rotations
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    
    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow("rotated image", rotated) 

#resizing
resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
cv.imshow("resized images", resized)

#flipping
'''
0 -> flipped vertically
1 -> flipped horizontally
-1 -> flipped both horizontally and vertically
'''
flipped = cv.flip(img, 0)
cv.imshow('flipped images', flipped)

#cropping
cropped = img[200:300, 300:400]
cv.imshow('cropped images', cropped)

cv.waitKey(0)