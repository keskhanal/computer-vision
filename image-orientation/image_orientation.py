import os
import cv2
import numpy as np

def show_img(image:np.ndarray) -> None:
    """Resizes and displays an image.
    Args:
        image (np.ndarray): A numpy array representing the image.
    """
    # Get the original image dimensions
    height, width = image.shape[:2]

    # Set the new dimensions
    new_width = 600
    new_height = int(new_width * height / width)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Display the resized image
    cv2.imshow("test", resized_image)

    # Wait for a key press and then destroy the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_face(image, debug=False) -> np.ndarray or None:
    """Detects a face in an image using the OpenCV deep neural network (DNN) module.

    Args:
        image (numpy.ndarray): The input image to detect the face from.
        debug (bool, optional): A flag to enable/disable debug mode. Defaults to True.

    Returns:
        numpy.ndarray: The input image with a bounding box drawn around the detected face, if found.
                       If no face is detected or the confidence is too low, returns None.
    """
    # Load the pre-trained Caffe model
    base_path = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(base_path, "./model/deploy.prototxt.txt")
    model_path = os.path.join(base_path, "./model/res10_300x300_ssd_iter_140000.caffemodel")
    
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Create a blob from the image for input to the DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    net.setInput(blob)

    # Perform face detection using the DNN
    faces = net.forward()
    
    # If no faces are detected, return None
    if faces is None or len(faces) == 0:
        print("no face detected...")
        return None
    
    # Get the bounding box coordinates of the first detected face
    box = faces[0, 0, 0, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    # box[1] = box[1]+0.2*(box[3]-box[1])
    (x1, y1, x2, y2) = box.astype("int")
    (x, y, w, h) = (x1, y1, x2-x1, y2-y1)
    
    # Get the confidence score of the first detected face
    confidence = round(float(faces[0, 0, 0, 2]), 4)
    
    # If the confidence is too low or the face is too wide, return None
    if confidence < 0.5 or w > h:
        # show_img(image)
        print(f"confidence is too low i.e {confidence}")    
        return None

     # If debug mode is enabled, draw a bounding box and the confidence score on the image
    if debug:
        new_img = image.copy()
        cv2.rectangle(new_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = "{:.2f}%".format(confidence * 100)
        text_width, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_pos = (x1 + int((x2-x1)/2) - int(text_width/2), y1 - 10)
        cv2.putText(new_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        show_img(new_img)
    
    #return the original image
    print(f"confidence after rotation: {confidence}")    
    return image


def rotate_image(image:np.ndarray, center:tuple[int, int], scale:float, angle:float) -> np.ndarray:
    """Rotates an image around a specified center point, with a given scaling factor and angle.
    Args:
        image: A numpy array representing the image to rotate.
        center: A tuple containing the (x, y) coordinates of the center point around which to rotate the image.
        scale: A scaling factor applied to the image during rotation.
        angle: The angle (in degrees) to rotate the image by.

    Returns:
        np.ndarray: A numpy array representing the rotated image.
    """
    (h, w) = image.shape[:2]

    # Define the size of the output image
    cos_theta = abs(np.cos(np.deg2rad(angle)))
    sin_theta = abs(np.sin(np.deg2rad(angle)))
    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Adjust the rotation matrix to take into account the translation
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Apply the rotation and detect the face in the new image
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
    return detect_face(rotated_image, debug=True)
    

def main() -> None: 
    """Loads an image, detects a face in it, rotates the image by 90, 180, and 270 degrees, and 
    saves the first successful rotation as a new image.
    """
    try:
        image = cv2.imread("./test_img/2.jpg")
        angles = [90, 180, 270]
        
        rotated_image = detect_face(image)

        if rotated_image is None:
            for angle in angles:
                # calculate the center of the image
                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                scale = 1.0

                # rotate the image
                rotated_image = rotate_image(image, center, scale, angle)
                
                # If the rotation was successful, save the image and break out of the loop
                if rotated_image is not None:
                    break
        
        cv2.imwrite("./test_img/rotated.jpg", rotated_image)
    except Exception as e:
        print(f"error {str(e)} occured while rotation...")

if __name__ == "__main__":
    main()