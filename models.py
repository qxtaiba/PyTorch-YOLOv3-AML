from utils.utils import *
import torch.nn.functional as F
import os
import numpy as np
import math
import time
from copy import deepcopy
import torch
import torch.nn as nn

def parseModel(path):
    # init empty lists
    moduleDefinitions, validLines = [], []
    # read cfg file line by line and store it
    allLines = open(path, 'r').read().split('\n')
    
    for line in allLines:
        # check if line is not empty and do not start with '#'
        if line and not line.startswith("#"):
            # append line and strip all fringe whitespace 
            validLines.append(line.rstrip().lstrip())

    for line in validLines:
        # check if we are at the start of a new block 
        isNewBlock = line.startswith('[')
        
        if isNewBlock:
            # append and populate a dictionary to moduleDefinitions
            moduleDefinitions.append({})
            moduleDefinitions[-1]['type'] = line[1:-1].rstrip()
            # check if module type is convolutional and add batch norm parameter
            if moduleDefinitions[-1]['type'] == 'convolutional':
                # pre-populate with zeros (may be overwritten later)
                moduleDefinitions[-1]['batch_normalize'] = 0  
        
        else:
            # extract key, value pair
            key, val = line.split("=")
            # strip whitespace 
            key = key.rstrip()

            # return a numpy array 
            if key == 'anchors':  
                moduleDefinitions[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            # return a regular array 
            elif (key in ['from', 'layers', 'mask']):  
                moduleDefinitions[-1][key] = [int(x) for x in val.split(',')]
            # return a regular array 
            elif (key == 'size' and ',' in val): 
                moduleDefinitions[-1][key] = [int(x) for x in val.split(',')]

            else:
                # strip whitespace 
                val = val.strip()
                # return int/float 
                if val.isnumeric():
                    moduleDefinitions[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)   # return int or float
                # return string 
                else:
                    moduleDefinitions[-1][key] = val  

    return moduleDefinitions

def parseData(path):
    # init output dictionary 
    options = dict()

    # open are read data file into lines 
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # strip whitespace 
        line = line.strip()
        # check if line is empty or starts with a '#' (indicates a comment)
        if line == '' or line.startswith('#'): continue
        # extract key, value pair 
        key, val = line.split('=')
        # add key,value pair to dictionary 
        options[key.strip()] = val.strip()

    return options

# constructs module list of layer blocks from module configuration in moduleDefinitions
def createModules(moduleDefinitions, imgSize, cfg):
    
    # check if image size is an integer or tuple, and expand it if necessary 
    imgSize = [imgSize] * 2 if isinstance(imgSize, int) else imgSize  
    # extract hyperparameters from config file (unused)
    trainingHyperparms = moduleDefinitions.pop(0)  
    # init output filters
    outputFilters = [3]  
    # init module list 
    moduleList = nn.ModuleList()
    # init routing layers (list of layers that route to deeper layers)
    routingLayers = []  
    # init yolo index
    yoloIndex = -1

    # iterate through parsed cfg file and construct modules 
    for idx, currModule in enumerate(moduleDefinitions):
        modules = nn.Sequential()

        if currModule['type'] == 'convolutional':
            # extract batch normalize value 
            isBatchNormalize = currModule['batch_normalize']
            # extract filters value 
            filters = currModule['filters']
            # extract kernel size value
            kernelSize = currModule['size']   
            # extract stride value
            stride = currModule['stride'] if 'stride' in currModule else (currModule['stride_y'], currModule['stride_x'])
            # create convolutional layer 
            convLayer = nn.Conv2d(in_channels = outputFilters[-1], out_channels = filters, kernel_size = kernelSize, stride = stride, padding = kernelSize // 2 if currModule['pad'] else 0, groups = currModule['groups'] if 'groups' in currModule else 1, bias = not isBatchNormalize)
            # add convolutional layer 
            modules.add_module('Conv2d', convLayer)

            # if batch normalise 
            if isBatchNormalize:
                # create batch norm layer 
                batchNormLayer = nn.BatchNorm2d(filters, momentum = 0.03, eps = 1E-4)
                # add batch norm layer 
                modules.add_module('BatchNorm2d', batchNormLayer)
            
            else:
                # detection output that goes into YOLO layer 
                routingLayers.append(idx)  

            # leaky activation
            if currModule['activation'] == 'leaky':  
                # creat leaky layer 
                leakyLayer = nn.LeakyReLU(0.1, inplace = True)
                # add leaky layer 
                modules.add_module('activation', leakyLayer)

        elif currModule['type'] == 'upsample':
            # create upsample layer 
            modules = nn.Upsample(scale_factor = currModule['stride'])

        elif currModule['type'] == 'route':  
            # extract layers 
            layers = currModule['layers']
            # extract filters 
            filters = sum([outputFilters[l + 1 if l > 0 else l] for l in layers])
            # extend routing layers
            routingLayers.extend([idx + l if l < 0 else l for l in layers])
            # creat route layer using FeatureConcat class
            modules = FeatureConcat(layers = layers)

        elif currModule['type'] == 'shortcut':
            # extract layers 
            layers = currModule['from']
            # extract filters 
            filters = outputFilters[-1]
            # extend routing layers 
            routingLayers.extend([idx + l if l < 0 else l for l in layers])
            # create shortcut layer using WeightedFeatureFusion class
            modules = WeightedFeatureFusion(layers = layers, weight ='weights_type' in currModule)

        elif currModule['type'] == 'yolo':
            # increment yolo index 
            yoloIndex += 1
            # init stride list 
            stride = [32, 16, 8]
            # extract layers 
            layers = currModule['from'] if 'from' in currModule else []
            # create yolo layer 
            modules = YOLOLayer(anchors = currModule['anchors'][currModule['mask']], numClasses = currModule['classes'],   imageSize = imgSize,   yoloLayerIndex = yoloIndex,   layers = layers, stride = stride[yoloIndex])

            # init preceding Conv2d() bias 
            j = layers[yoloIndex] if 'from' in currModule else -1
            bias_ = moduleList[j][0].bias  
            bias = bias_[:modules.numOutputs * modules.numAnchors].view(modules.numAnchors, -1)  
            bias[:, 4] += -4.5  
            bias[:, 5:] += math.log(0.6 / (modules.numClasses - 0.99))  
            moduleList[j][0].bias = torch.nn.Parameter(bias_, requires_grad = bias_.requires_grad)

        # append modules to module list 
        moduleList.append(modules)
        # append filters to output filters 
        outputFilters.append(filters)
    
    # init binary routing layers 
    binaryRoutingLayers = [False] * (idx + 1)
    # iterate through routing layers and set those indices to true in binary routing layers list 
    for idx in routingLayers:
        binaryRoutingLayers[idx] = True

    return moduleList, binaryRoutingLayers


# parses and loads the weights stored in 'weights'
def loadDarkNetWeights(self, weights, cutoff=-1):

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for idx, (moduleDef, module) in enumerate(zip(self.moduleDefinitions[:cutoff], self.moduleList[:cutoff])):
        if moduleDef['type'] == 'convolutional':
            # extract conv
            conv = module[0]

            # load batch normalization bias, weights, running mean and running variance
            if moduleDef['batch_normalize']:

                # extract batch normalize 
                batchNormalize = module[1]

                # extract number of biases
                numBiases = batchNormalize.bias.numel() 

                # load bias
                batchNormBias = torch.from_numpy(weights[ptr:ptr + numBiases])
                
                batchNormalize.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + numBiases]).view_as(batchNormalize.bias))
                ptr += numBiases

                # load weight
                batchNormWeight = torch.from_numpy(weights[ptr:ptr + numBiases])
                
                batchNormalize.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + numBiases]).view_as(batchNormalize.weight))
                ptr += numBiases

                # load running mean 
                batchNormRunMean = torch.from_numpy(weights[ptr:ptr + numBiases])

                batchNormalize.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + numBiases]).view_as(batchNormalize.running_mean))
                ptr += numBiases

                # load running var 
                batchNormRunVar = torch.from_numpy(weights[ptr:ptr + numBiases])

                batchNormalize.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + numBiases]).view_as(batchNormalize.running_var))
                ptr += numBiases

                # cast into dimensions of the model 
                batchNormBias = batchNormBias.view_as(batchNormalize.bias)
                batchNormWeight = batchNormWeight.view_as(batchNormalize.weight)
                batchNormRunMean = batchNormRunMean.view_as(batchNormalize.running_mean)
                batchNormRunVar = batchNormRunVar.view_as(batchNormalize.running_var)

                # copy data to model 
                batchNormalize.bias.data.copy_(batchNormBias)
                batchNormalize.weight.data.copy_(batchNormWeight)
                batchNormalize.running_mean.data.copy_(batchNormRunMean)
                batchNormalize.running_var.data.copy_(batchNormRunVar)

            else:
                # extract number of biases 
                numBiases = conv.bias.numel()
                # load bias 
                convBias = torch.from_numpy(weights[ptr:ptr + numBiases])
                # cast into dimensions of model
                convBias = convBias.view_as(conv.bias)
                # copy data to model 
                conv.bias.data.copy_(convBias)
                # increment pointer 
                ptr += numBiases

            # load conv weights
            # extract number of weights 
            numWeights = conv.weight.numel()  
            # load weights 
            convWeights = torch.from_numpy(weights[ptr:ptr + numWeights])
            # cast into dimensions of model 
            convWeights = convWeights.view_as(conv.weight)
            # copy data to model 
            conv.weight.data.copy_(convWeights)
            # increment pointer 
            ptr += numWeights


class YOLOLayer(nn.Module):
    def __init__(self, anchors, numClasses, imageSize, yoloLayerIndex, layers, stride):
        super(YOLOLayer, self).__init__()

        # init class variables
        self.anchors = torch.Tensor(anchors)
        self.layerIndex = yoloLayerIndex  
        self.layerIndices = layers  
        self.layerStride = stride  
        self.numOutputLayers = len(layers)  
        self.numAnchors = len(anchors) 
        self.numClasses = numClasses  
        self.numOutputs = numClasses + 5  
        self.numXGridPoints, self.numYGridPoints, self.numGridpoints = 0, 0, 0  
        self.anchorVector = self.anchors / self.layerStride
        self.anchorWH = self.anchorVector.view(1, self.numAnchors, 1, 1, 2)


    def creatGrids(self, numGridPoints =(13, 13), device ='cpu'):
        
        # extract number of x, y gridpoints
        self.numXGridPoints, self.numYGridPoints = numGridPoints  
        # create gridpoints tensor
        self.numGridpoints = torch.tensor(numGridPoints, dtype = torch.float)

        # check if not currently training and build xy offsets 
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.numYGridPoints, device = device), torch.arange(self.numXGridPoints, device = device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.numYGridPoints, self.numXGridPoints, 2)).float()

        # check if devices do not match and send to device 
        if self.anchorVector.device != device:
            self.anchorVector = self.anchorVector.to(device)
            self.anchorWH = self.anchorWH.to(device)

    def forward(self, prediction, out):
        
        # extract batch size, number of y gridpoints, number of x gridpoints
        batchSize, _, numYGridPoints, numXGridPoints = prediction.shape  

        # check if there is a mismatch in grid sizes and create grids
        if (self.numXGridPoints, self.numYGridPoints) != (numXGridPoints, numYGridPoints):
            self.creatGrids((numXGridPoints, numYGridPoints), prediction.device)

        # reshape prediction accordingly 
        prediction = prediction.view(batchSize, self.numAnchors, self.numOutputs, self.numYGridPoints, self.numXGridPoints).permute(0, 1, 3, 4, 2).contiguous()  

        # check if training is true
        if self.training:
            return prediction
        
        #inference 
        else:
            # extract inference output
            inferenceOutput = prediction.clone() 
            # xy calculation
            inferenceOutput[..., :2] = torch.sigmoid(inferenceOutput[..., :2]) + self.grid  
            # wh yolo method calculation
            inferenceOutput[..., 2:4] = torch.exp(inferenceOutput[..., 2:4]) * self.anchorWH  
            # multiply xywh by layer stride 
            inferenceOutput[..., :4] *= self.layerStride
            # pass inferenceOutput[..., 4:] through sigmoid function
            torch.sigmoid_(inferenceOutput[..., 4:])

            # view [1, 3, 13, 13, 85] as [1, 507, 85]
            return inferenceOutput.view(batchSize, -1, self.numOutputs), prediction  


# YOLOv3 object detection model
class Darknet(nn.Module):

    def __init__(self, cfg, imageSize =(416, 416), verbose = False):
        super(Darknet, self).__init__()

        # init class variables
        self.moduleDefinitions = parseModel(cfg)
        self.moduleList, self.routs = createModules(self.moduleDefinitions, imageSize, cfg)
        self.yoloLayers = [i for i, m in enumerate(self.moduleList) if m.__class__.__name__ == 'YOLOLayer']  
        self.version = np.array([0, 2, 5], dtype = np.int32)  
        self.numImageSeen = np.array([0], dtype = np.int64)  

    def forward(self, x, augment = False, verbose = False):
        
        # check if augment is false 
        if not augment:
            # pass x through Once
            return self.forwardOnce(x)

        else:  
            # extract image size 
            imageSize = x.shape[-2:]  
            # init scales
            scales = [0.83, 0.67] 
            # init y list  
            output = []

            # iterate through x, flipped (left-right) and scaled x, scaled x 
            for i, xi in enumerate((x, torch_utils.scaleImage(x.flip(3), scales[0], same_shape = False),  torch_utils.scaleImage(x, scales[1], same_shape = False))):
                # pass value through forward once and append output
                output.append(self.forwardOnce(xi)[0])

            # scale
            output[1][..., :4] /= scales[0]  
            # flip left-right
            output[1][..., 0] = imageSize[1] - output[1][..., 0]  
            # scale
            output[2][..., :4] /= scales[1]

            # concatenate output
            output = torch.cat(output, 1)

            return output, None

    def forwardOnce(self, inferenceOutput):

        # init list for yolo output and output 
        yoloLayerOutput, output = [], []

        # iterate through modules in module list
        for i, module in enumerate(self.moduleList):
            # extract module class name 
            name = module.__class__.__name__

            # check if module is of type WeightedFeatureFusion or FeatureConcat
            if name in ['WeightedFeatureFusion', 'FeatureConcat']: 
                # extract inference output 
                inferenceOutput = module(inferenceOutput, output)  
            
            # check if module is of type YOLOLayer
            elif name == 'YOLOLayer':
                # extract and append yolo output 
                yoloLayerOutput.append(module(inferenceOutput, output))

            else: 
                # extract inference output
                inferenceOutput = module(inferenceOutput)

            # append inference output to output list 
            output.append(inferenceOutput if self.routs[i] else [])

        # check if training 
        if self.training:
            return yoloLayerOutput
        
        # inference or test
        else: 
            # extract inference and training output 
            inferenceOutput, trainingOutput = zip(*yoloLayerOutput)  
            # concatenate yolo outputs 
            inferenceOutput = torch.cat(inferenceOutput, 1)  

            return inferenceOutput, trainingOutput

    # Fuse Conv2d + BatchNorm2d layers throughout model
    def fuse(self):
        # init instance of nn.ModuleList()
        fuseList = nn.ModuleList()
        
        # iterate through child modules
        for child in list(self.children())[0]:
           
            # check if child is of type nn.Sequential
            if isinstance(child, nn.Sequential):
           
                # iterate throguh child
                for index, bn in enumerate(child):
           
                    # check if current val is of type nn.modules.batchnorm.BatchNorm2d
                    if isinstance(bn, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = child[index - 1]
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

                        child = nn.Sequential(fusedconv, *list(child.children())[index + 1:])
                        break
            
            # append child to fused list 
            fuseList.append(child)
        # assign module list to fuse list 
        self.moduleList = fuseList
 
class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layerIndices = layers  
        self.isMultipleLayers = len(layers) > 1  

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layerIndices], 1) if self.isMultipleLayers else outputs[self.layerIndices[0]]


# weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
class WeightedFeatureFusion(nn.Module):  
    def __init__(self, layers, weight = False):
        super(WeightedFeatureFusion, self).__init__()
        self.layerIndices = layers  
        self.isApplyWeights = weight  
        self.numLayers = len(layers) + 1 

        if weight:
            self.layerWeights = nn.Parameter(torch.zeros(self.numLayers), requires_grad = True)  

    def forward(self, x, outputs):
        if self.isApplyWeights:
            w = torch.sigmoid(self.layerWeights) * (2 / self.numLayers)  
            x = x * w[0]

        inputChannels = x.shape[1]  
        
        for i in range(self.numLayers - 1):
            addFeatures = outputs[self.layerIndices[i]] * w[i + 1] if self.isApplyWeights else outputs[self.layerIndices[i]]  
            featureChannles = addFeatures.shape[1]  

            if inputChannels == featureChannles:
                x = x + addFeatures
            elif inputChannels > featureChannles:  
                x[:, :featureChannles] = x[:, :featureChannles] + addFeatures  
            else:
                x = x + addFeatures[:, :inputChannels]

        return x
