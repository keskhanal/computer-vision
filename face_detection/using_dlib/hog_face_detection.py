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

detector = dlib.get_frontal_face_detector()

# BGR to RGB channel ordering (which is what dlib expects)
img_path = "./test_img/pkd1.jpg"
image = cv2.imread(img_path)
image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# perform face detection using dlib's face detector
start = time.time()

print("[INFO[ performing face detection with dlib...")
rects = detector(rgb, 2)
rects = [convert_and_trim_bb(image, r) for r in rects]
show_img(image, rects)

end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))