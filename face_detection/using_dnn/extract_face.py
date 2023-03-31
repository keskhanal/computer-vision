import cv2
import numpy as np

# Display the result
def show_image(img:np.ndarray, x:int, y:int, w:int, h:int, conf:float)->None:
    #draw bounding box around the face and show image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    text = "{:.2f}%".format(conf * 100)
    text_width, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    text_pos = (x + int(w/2) - int(text_width/2), y - 10)
    cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    cv2.imshow("Face detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_blur(image:np.ndarray)->float:
    """computes laplacian and returns the focus measure(ie varience for the image)
    args:
        image(np.ndarray): cv2 image
    returns:
        fm(float): focus measure of image(variance of laplacian)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

# ====================== face detection with dnn =============================================
def detect_face(image, debug=False)->tuple:
    # Load DNN-based face detector
    prototxt_path = "./detect_face/models/deploy.prototxt.txt"
    model_path = "./detect_face/models/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    #set the input blob for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)

    #detect face
    detections = net.forward()
    if detections is None or len(detections) == 0:
        return None, None, 0.0
    
    confidence = round(float(detections[0, 0, 0, 2]), 4)
    if confidence < 0.5:    
        return None, None, 0.0
    
    box = detections[0, 0, 0, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    box[1] = box[1]+0.2*(box[3]-box[1])
    (x1, y1, x2, y2) = box.astype("int")
    (x, y, w, h) = (x1, y1, x2-x1, y2-y1)
    bounding_box={"top": int(x), "left": int(y), "width": int(w), "height": int(h)}
    
    face = image[y:y+h, x:x+w]
    # if (w, h) != required_size:
    #     face = cv2.resize(face, required_size, interpolation=cv2.INTER_LINEAR)

    if debug:
        show_image(image, x, y, w, h, confidence)
    
    return face, bounding_box, confidence

if __name__ == "__main__":
    image = cv2.imread("../test_img/pp.jpeg")
    detect_face(image, debug=True)