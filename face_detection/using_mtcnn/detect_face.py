import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import cv2
import numpy as np
from mtcnn import MTCNN

# Display the result
def show_image(image:np.ndarray, bbox:list, keypoints, conf:float):
    #draw bounding box around the face and show image
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

    text = "{:.2f}%".format(conf * 100)
    text_width, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    text_pos = (bbox[0] + int(bbox[2]/2) - int(text_width/2), bbox[1] - 10)
    cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    cv2.imshow("Face detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_blur(image:np.ndarray):
    """computes laplacian and returns the focus measure(ie varience for the image)
    args:
        image(np.ndarray): cv2 image
    returns:
        fm(float): focus measure of image(variance of laplacian)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


# ====================== face detection with mtcnn =============================================
def extract_face(image:np.ndarray, debug:bool=False)->tuple:
    # Load MTCNN face detector
    detector = MTCNN()

    #detect faces in the image
    results = detector.detect_faces(image)
    
    if not results:
        return None, None, 0.0

    face_data = results[0]
    confidence = round(float(face_data["confidence"]), 4)
    keypoints = face_data["keypoints"]
    bbox = face_data['box']
    bounding_box={"top": bbox[0], "left": bbox[1], "width":bbox[2], "height":bbox[3]}

    face = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    # if (w, h) != required_size:
    #     face = cv2.resize(face, required_size, interpolation=cv2.INTER_LINEAR)

    if debug:
        show_image(image, bbox, keypoints, confidence)
    
    return face, bounding_box, confidence


if __name__ == "__main__":
    img = cv2.imread("./test_img/pp.jpeg")
    detected_face, bounding_box, conf = extract_face(img, debug=1)
