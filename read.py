import cv2 as cv

#reading images
# img = cv.imread('images/space.jpg')
# cv.imshow('space', img)

#cv.waitKey(0)


#reading videos
capture = cv.VideoCapture('videos/Sand.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('sand in a hand',frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()