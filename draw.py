import cv2 as cv
import numpy as np

#for blank images
blank = np.zeros((500,500, 3), dtype='uint8')
cv.imshow("blank", blank)

'''
#paint a blank images with certain color
blank[200:300, 300:400] = 0,255,0
cv.imshow("Green", blank)

#draw a rectangle
cv.rectangle(blank, (0,0), (blank.shape[0]//2,blank.shape[1]//2), (0,0,255), thickness=1)
cv.imshow("rectangle", blank)

#draw a circle
cv.circle(blank, (blank.shape[0]//2,blank.shape[1]//2), 40, (0,255,0), thickness=2)
cv.imshow("circle", blank)

#draw line
cv.line(blank, (200,100), (blank.shape[0]//2,blank.shape[1]//2), (255,255,255), thickness=2)
cv.imshow("line", blank)
'''

#put text to the image
cv.putText(blank, "hello", (250, 250), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), thickness=2)
cv.imshow("Text", blank)

cv.waitKey(0)