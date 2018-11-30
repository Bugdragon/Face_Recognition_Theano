import os
import sys
import time
import numpy as np
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

# 图像加载函数
def load_data(dataset_path):
    img = Image.open(dataset_path)
    img_ndarray = np.asarray(img, dtype='float64')/256
    
    faces = np.empty((400,2679)) # 400张人脸图像，每张共2679个像素
    for row in range(20):
        for column in range(20):
            faces[]
