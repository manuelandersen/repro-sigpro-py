import sys
import os
sys.path.append(os.path.abspath('../utilities'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.color import rgb2gray
from skimage import io
import pywt
from PIL import Image

from function_huber_l2 import function_huber_l2
from grad_huberl2 import grad_huber_l2
from mask_param import mask_param

# Define colors
hgrad = np.array([0, 0, 204]) / 255
hfb2 = np.array([153, 0, 153]) / 255
hfb3 = np.array([255, 51, 153]) / 255
hpr = np.array([100, 29, 29]) / 255
hdr = np.array([192, 135, 70]) / 255
col = plt.cm.hsv(np.linspace(0, 1, 7))

# Load and process image
img = Image.open('IMG_5885.jpg').convert('L')
x0 = np.array(img, dtype=np.float64)
x = x0[762:888:2, 923:1112:3]
sx = np.sqrt(x.size)

#TODO: no se cual de las dos es la manera correcta
pywt.Modes.from_object('periodization')  # Equivalent to MATLAB's dwtmode('per')
#pywt.Modes.modes['per'] = pywt.Modes.MODE_PERIODIC

#TODO: averiguar que configura esto
prox = {'rm': 3}
prox['wav'] = 'sym3' 

print(prox)
print(prox['rm'])
      
print(np.log2(sx))
print(type(np.log2(sx)))
N = int(np.log2(sx))
print(N)
print(type(N))
mask = mask_param(np.log2(sx), prox.rm, prox.wav) + 1e-16
print(mask)