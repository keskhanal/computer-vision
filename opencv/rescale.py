import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    #for images, videos and live videos
    width = int(frame.shape[0]*scale)
    height = int(frame.shape[1]*scale)
    
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#reading images
img = cv.imread('images/space.jpg')

cv.imshow('space', img)

resized_img = rescaleFrame(img,  scale=0.2)
cv.imshow("resized image", resized_img)

cv.waitKey(0)

"""
#reading videos
capture = cv.VideoCapture('videos/Sand.mp4')

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame, scale=0.2)

    cv.imshow('sand in a hand',frame)
    cv.imshow('resized video', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
"""
