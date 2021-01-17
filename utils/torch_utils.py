import math
import os
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F


def fuseConvBnLayers(conv, bn):

    # disable gradient calculation
    with torch.no_grad():
        # crate fused convolutional layer 
        fusedconv = torch.nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size = conv.kernel_size, stride = conv.stride, padding = conv.padding, bias = True)
        # extract convolutional weights
        convolutionalWeights = conv.weight.clone().view(conv.out_channels, -1)
        # extract batch normalization weights 
        batchNormWeights = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        # init and reshape fused convolutional layer weights 
        fusedconv.weight.copy_(torch.mm(batchNormWeights, convolutionalWeights).view(fusedconv.weight.size()))

        # check if convolutional layer bias is not none 
        if conv.bias is not None:
            # extract convolutional spatial bias 
            convolutionalBias = conv.bias
        else:
            # set to zero tensor 
            convolutionalBias = torch.zeros(conv.weight.size(0))
        
        # extract batch normalization spatial bias
        batchNormBias = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        # init and reshape fused convolutional layer bias
        fusedconv.bias.copy_(torch.mm(batchNormWeights, convolutionalBias.reshape(-1, 1)).reshape(-1) + batchNormBias)

        return fusedconv

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
