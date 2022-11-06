import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (50, 50), (350, 350), 255, -1)
cv.imshow("rectangle", rectangle)

circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
cv.imshow("circle", circle)

#bitwise AND --> intersecting region
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow("bitwise_and", bitwise_and)

#bitwise OR -- both intersecting and non intersecting region
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow("bitwise_or", bitwise_or)

#bitwise XOR --> non intersecting region
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('bitwise_xor', bitwise_xor)

#bitwise NOT --> inversing color
bitwise_not = cv.bitwise_not(circle)
cv.imshow("bitwise_not", bitwise_not)

cv.waitKey(0)