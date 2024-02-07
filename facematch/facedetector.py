from typing import Any, Union
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
import time
import cv2
import dlib
# Model's weights paths
#instead of cwd get the absolute current path
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolov8n-face.pt")
PATH_DLIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "dlib_face_recognition_resnet_model_v1.dat")
# Google Drive URL
#"https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"

# Confidence thresholds for landmarks detection
# used in alignment_procedure function
LANDMARKS_CONFIDENCE_THRESHOLD = 0.5

yolo_model = YOLO(PATH)
detector = dlib.get_frontal_face_detector()

def detect_face(
    face_detector: Any, img: np.ndarray, align: bool = True
) -> tuple:
    """
    Detect a single face from a given image
    Args:
        face_detector (Any): pre-built face detector object
        detector_backend (str): detector name
        img (np.ndarray): pre-loaded image
        alig (bool): enable or disable alignment after detection
    Returns
        result (tuple): tuple of face (np.ndarray), face region (list)
            , confidence score (float)
    """
    obj = detect_faces(face_detector, img, align)

    if len(obj) > 0:
        face, region, confidence = obj[0]  # discard multiple faces

    # If no face is detected, set face to None,
    # image region to full image, and confidence to 0.
    else:  # len(obj) == 0
        face = None
        region = [0, 0, img.shape[1], img.shape[0]]
        confidence = 0

    return face, region, confidence


def detect_faces(
    face_detector: Any, img: np.ndarray, align: bool = True
) -> list:
    """
    Detect face(s) from a given image
    Args:
        face_detector (Any): pre-built face detector object
        detector_backend (str): detector name
        img (np.ndarray): pre-loaded image
        alig (bool): enable or disable alignment after detection
    Returns
        result (list): tuple of face (np.ndarray), face region (list)
            , confidence score (float)
    """

    detect_face_fn = detect_face_yolo
    obj = detect_face_fn(yolo_model, img, align)
    
    return obj
    



def alignment_procedure(
    img: np.ndarray, left_eye: Union[list, tuple], right_eye: Union[list, tuple]
) -> np.ndarray:
    """
    Rotate given image until eyes are on a horizontal line
    Args:
        img (np.ndarray): pre-loaded image
        left_eye: coordinates of left eye with respect to the you
        right_eye: coordinates of right eye with respect to the you
    Returns:
        result (np.ndarray): aligned face
    """
    angle = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    img = Image.fromarray(img)
    img = np.array(img.rotate(angle))
    return img



def find_best_angle(image):
    degrees = np.arange(-90,90 + 45,45)#
    #drop 0 if in degree
    degrees = degrees[degrees != 0]
    image = cv2.resize(image, (128,128))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) > 0:
        return 0

    for degree in degrees:
        rows,cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
        image_ = cv2.warpAffine(image,M,(cols,rows))
        gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        if len(faces) > 0:
            break

    return degree

def detect_face_yolo(face_detector: Any, img: np.ndarray, align: bool = False) -> list:
    """
    Detect and align face with yolo
    Args:
        face_detector (Any): yolo face detector object
        img (np.ndarray): pre-loaded image
        align (bool): default is true
    Returns:
        list of detected and aligned faces
    """
    resp = []
    # Detect faces

    if len(np.shape(img)) == 4:
        img = img[0]

    degree = find_best_angle(img)
    if degree != 0:
        rows,cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
        img = cv2.warpAffine(img,M,(cols,rows))
        
    #print(degree)

    results = face_detector.predict(img, verbose=False, show=False, conf=0.1)[0]

    # For each face, extract the bounding box, the landmarks and confidence
    for result in results:

        # Extract the bounding box and the confidence
        x, y, w, h = result.boxes.xywh.tolist()[0]
        confidence = result.boxes.conf.tolist()[0]

        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
        detected_face = img[y : y + h, x : x + w].copy()

        if align:
            # Tuple of x,y and confidence for left eye
            left_eye = result.keypoints.xy[0][0], result.keypoints.conf[0][0]
            # Tuple of x,y and confidence for right eye
            right_eye = result.keypoints.xy[0][1], result.keypoints.conf[0][1]

            # Check the landmarks confidence before alignment
            if (
                left_eye[1] > LANDMARKS_CONFIDENCE_THRESHOLD
                and right_eye[1] > LANDMARKS_CONFIDENCE_THRESHOLD
            ):
                detected_face = alignment_procedure(
                    detected_face, left_eye[0].cpu(), right_eye[0].cpu()
                )
        resp.append((detected_face, [x, y, w, h], confidence))

    return resp


##################################################################################
##################################################################################
##################################################################################
##################################################################################
