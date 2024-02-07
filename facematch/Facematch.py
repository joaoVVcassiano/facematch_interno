# configurations for dependencies
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# -----------------------------------
# common dependencies
import time
import logging
from typing import Any, Dict, List, Tuple, Union

# 3rd party dependencies
import numpy as np
import tensorflow as tf

from . import functions
from . import distance as dst
from . import vgg_face
from . import facenet
# -----------------------------------

#model = vgg_face.loadModel()
model = facenet.loadModel()

def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    align: bool = True,
) -> Dict[str, Any]:

    tic = time.time()

    # --------------------------------
    target_size = (160,160)

    # img pairs might have many faces
    img1_objs = functions.extract_faces(
        img=img1_path,
        target_size=target_size,
        grayscale=False,
        align=align,
    )

    img2_objs = functions.extract_faces(
        img=img2_path,
        target_size=target_size,
        grayscale=False,
        align=align,
    )
    # --------------------------------
    distances = []
    regions = []
    confidences = []
    # now we will find the face pair with minimum distance
    for img1_content, img1_region, confidence1 in img1_objs:
        for img2_content, img2_region, confidence2 in img2_objs:
            img1_embedding_obj = represent(
                img_path=img1_content,
            )

            img2_embedding_obj = represent(
                img_path=img2_content,
            )

            img1_representation = img1_embedding_obj[0]["embedding"]
            img2_representation = img2_embedding_obj[0]["embedding"]

            distance = dst.findCosineDistance(img1_representation, img2_representation)

            distances.append(distance)
            regions.append((img1_region, img2_region))
            confidences.append((confidence1, confidence2))

    if len(distances) == 0:
        
        toc = time.time()

        resp_obj = {
            "similarity": 0,
            "facial_areas": {"img1": None, "img2": None},
            "time": round(toc - tic, 2),
            "confidence": {'r_image' : None, 'c_image': None},}

        return resp_obj
    
    arg_min = np.argmin(distances)

    confidence = confidences[arg_min]
    distance = min(distances)  # best distance
    distance = distance if distance <= 1 else 1   # make sure the distance is in range [0,1]
    facial_areas = regions[np.argmin(distances)]

    toc = time.time()

    resp_obj = {
        "similarity": round(1 - distance,2),
        "facial_areas": {"r_image": facial_areas[0], "c_image": facial_areas[1]},
        "confidence": {'r_image' : round(confidence[0],2), 'c_image': round(confidence[1],2)},
        "time": round(toc - tic, 2),
    }

    return resp_obj

def represent(
    img_path: Union[str, np.ndarray],
) -> List[Dict[str, Any]]:


    resp_objs = []

    if len(img_path.shape) == 4:
        img_path = img_path[0]
    
    img_region = {"x": 0, "y": 0, "w": img_path.shape[1], "h": img_path.shape[2]}
    img_objs = [(img_path, img_region, 0)]


    for img, region, confidence in img_objs:
        # custom normalization
        img = functions.normalize_input(img=img)
        img = np.expand_dims(img, axis=0)
        embedding = model.predict(img, verbose =0)[0].tolist()
        #embedding = predict_tflite(interpreter, input_details, output_details, img)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs