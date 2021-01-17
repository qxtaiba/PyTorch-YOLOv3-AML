import math
import os
import time
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

def scaleImage(img, ratio = 1.0, same_shape = True):  # img(16,3,256,416), r = ratio
    
    # extract width and height 
    height, width = img.shape[2:]
    # calculate new size 
    newSize = (int(height * ratio), int(width * ratio))  
    # resize image
    img = F.interpolate(img, size = newSize, mode ='bilinear', align_corners = False)  
    
    # check if isSameShape set to false 
    if not same_shape:
        # calculate height and wdith to be used for padding/cropping image
        gridSize = 64  
        height, width = [math.ceil(x * ratio / gridSize) * gridSize for x in (height, width)]
        
    # pad image 
    return F.pad(img, [0, width - newSize[1], 0, height - newSize[0]], value = 0.447)  
