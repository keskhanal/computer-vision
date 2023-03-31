import time
import cv2
import dlib
import imutils

def convert_and_trim_bb(image, rect):
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	
    # dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	
    # compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	
    # return our bounding box coordinates
	return (startX, startY, w, h)

# process the detection boxes and draw them around faces
def show_img(img, rects):
    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)
    cv2.imshow('Result', img)
    cv2.waitKey(0)

#download dlib cnn model from this link "https://github.com/davisking/dlib-models/blob/master/mmod_human_face_detector.dat.bz2"
detector = dlib.cnn_face_detection_model_v1("./dlib_model/mmod_human_face_detector.dat")

# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)

image = cv2.imread("./test_img/pkd1.jpg")
image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# perform face detection using dlib's face detector
start = time.time()

print("performing face detection with dlib...")

results = detector(rgb, 1)
rects = [convert_and_trim_bb(image, r.rect) for r in results]

# show_img(image, rects)
print(rects)

end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))