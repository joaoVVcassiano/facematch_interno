import os
from typing import Union, Tuple
import base64
from pathlib import Path

# 3rd party dependencies
from PIL import Image
import numpy as np
import tensorflow as tf

from . import facedetector

import cv2

from tensorflow.keras.preprocessing import image

face_detector = facedetector.yolo_model

def extract_faces(
    img: Union[str, np.ndarray],
    target_size: tuple = (224, 224),
    grayscale: bool = False,
    align: bool = True,
) -> list:

    # this is going to store a list of img itself (numpy), it region and confidence
    extracted_faces = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img, img_name = img, "numpy array"

    face_objs = facedetector.detect_faces(face_detector, img, align)
    
    for current_img, current_region, confidence in face_objs:
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            if grayscale is True:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            # resize and padding
            factor_0 = target_size[0] / current_img.shape[0]
            factor_1 = target_size[1] / current_img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (
                int(current_img.shape[1] * factor),
                int(current_img.shape[0] * factor),
            )
            current_img = cv2.resize(current_img, dsize)

            diff_0 = target_size[0] - current_img.shape[0]
            diff_1 = target_size[1] - current_img.shape[1]
            if grayscale is False:
                # Put the base image in the middle of the padded image
                current_img = np.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                        (0, 0),
                    ),
                    "constant",
                )
            else:
                current_img = np.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                    ),
                    "constant",
                )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

            # normalizing the image pixels
            # what this line doing? must?
            img_pixels = image.img_to_array(current_img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            # int cast is for the exception - object of type 'float32' is not JSON serializable
            region_obj = {
                "x": int(current_region[0]),
                "y": int(current_region[1]),
                "w": int(current_region[2]),
                "h": int(current_region[3]),
            }

            extracted_face = [img_pixels, region_obj, confidence]
            extracted_faces.append(extracted_face)

    return extracted_faces


def normalize_input(img: np.ndarray) -> np.ndarray:

    #normalize image between 0 and 1
    img = img.astype("float32")
    #img = (img - img.min()) / (img.max() - img.min())

    #img *= 255

    #img[..., 0] -= 93.5940
    #img[..., 1] -= 104.7624
    #img[..., 2] -= 129.1863

    mean, std = img.mean(), img.std()
    img = (img - mean) / std

    return img

