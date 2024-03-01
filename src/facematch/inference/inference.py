import json
import logging
import os
from time import time
from datetime import datetime
import sys
import boto3
import numpy as np
from PIL import Image

#append path to ../.. to import facematch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from facematch import Facematch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def load_image(image_path):
    """
    input:
    - image_path : str : path to image
    
    output:
    - image as numpy array in bgr
    """


    #load image from s3
    s3 = boto3.client('s3')
    
    #bucket from path_image
    bucket = image_path.split('/')[0]
    key = '/'.join(image_path.split('/')[1:])
    
    response = s3.get_object(Bucket=bucket, Key=key)
    image = Image.open(response['Body'])
    image_bgr = np.array(image)
    return image_bgr


def input_fn(request_body, content_type):
    if content_type == 'application/json':
        request = json.loads(request_body)
        return request
    else:
        raise ValueError("Content type {} not supported.".format(content_type))
    

def predict_fn(input_data, model = None):
    logger.info('Predicting...')

    #load image path
    image_registry = input_data['image_registry']
    image_comparision = input_data['image_comparision']

    #load image from s3
    image_registry = load_image(image_registry)
    image_comparision = load_image(image_comparision)

    #call model for comparision
    result = Facematch.verify(image_registry, image_comparision)

    dict_result = {
        'similarity': result.get('similarity'),
        'time' : result.get('time'),
        'registry_image_confidence' : result.get('confidence').get('registry_image_confidence'),
        'comparision_image_confidence' : result.get('confidence').get('comparision_image_confidence'),
        'vec_embbeding_registry' : result.get('embbedings').get('vec_registry'),
        'vec_embbeding_ccomparision' : result.get('embbedings').get('vec_comparision')
        }
    
    return result