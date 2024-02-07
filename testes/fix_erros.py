import numpy as np
import cv2

import sys
sys.path.append('../')
from facematch import Facematch
#%load_ext autotime

r_image = '../base/imagens/f6a47b32-c401-4fca-b945-7a70f63292dd_selfie.jpeg'
c_image = '../base/imagens/f6a47b32-c401-4fca-b945-7a70f63292dd_transaction_selfie.jpeg'

r_image = cv2.imread(r_image)
c_image = cv2.imread(c_image)

result = Facematch.verify(r_image, c_image)
result = Facematch.verify(r_image, c_image)

print(result)