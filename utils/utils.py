import glob
import math
import os
import random
import shutil
import subprocess
import time
from copy import copy
from pathlib import Path
from sys import platform

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from . import torch_utils  

# set printoptions
torch.set_printoptions(linewidth = 320, precision = 5, profile ='long')
np.set_printoptions(linewidth = 320, formatter ={'float_kind': '{:11.5g}'.format})  # format short g, %precision = 5
matplotlib.rc('font', **{'size': 11})

# prevent OpenCV from multithreading in order to use PyTorch DataLoader
cv2.setNumThreads(0)

# convert boxes from [x1, y1, x2, y2] to [x, y, w, h] where x1, y1 represents the top-left coordinates and x2, y2 represents the bottom-right
def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # center x coordinate
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # center y coordinate
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

# convert boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1 = top-left, xy2 = bottom-right
def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top-left x coordinate
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top-left y coordinate
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom-right x coordinate
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom-right y coordinate
    return y


# rescale coordinates from first shape to that of second shape 
def scaleCoordinates(img1_shape, coords, img0_shape, ratio_pad = None):
    
    # check if ratioPad is set to None
    if ratio_pad is None:  
        # calculate gain (old/new) from img0_shape
        gain = max(img1_shape) / max(img0_shape) 
        # calculate width and height padding from img0_shape
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2 
    
    else:
        # calculate gain from ratioPad
        gain = ratio_pad[0][0]
        # calculate padding from ratioPad 
        pad = ratio_pad[1]

    # extract x and y padding valyes 
    xPadding, yPadding = pad[0], pad[1]

    # rescale coordinates
    coords[:, [0, 2]] -= xPadding 
    coords[:, [1, 3]] -= yPadding  
    coords[:, :4] /= gain

    # clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2

    return coords

# here we want to create a precision-recall curve and compute the average precision for each class
# F1 score is (harmonic mean of precision and recall)
def getAPClass(truePositives, objectnessVal, predictedClasses, targetClasses):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """


    # sort by objectness and store sorted indices in objectnessSortIndices
    objectnessSortIndices = np.argsort(-objectnessVal)
    # sort truePositive using sorted indices
    truePositives = truePositives[objectnessSortIndices]
    # sort objectnessVal using sorted indices
    objectnessVal = objectnessVal[objectnessSortIndices]
    # sort predClasses using sorted indices
    predictedClasses = predictedClasses[objectnessSortIndices]
    # find all unique classes 
    uniqueClasses = np.unique(targetClasses)
    # init constant val
    constVal = 1e-16

    # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    precisionScore = 0.1  
    shape = [uniqueClasses.shape[0], truePositives.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    AP, precision, recall = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    # iterate through each class stored in unique classes
    for classIndex, uniqueClass in enumerate(uniqueClasses):
        objectnessSortIndices = predictedClasses == uniqueClass
        # find number of ground truth objects
        numGroundTruthObjects = (targetClasses == uniqueClass).sum() 
        # find number of predicted objects 
        numPredictedObjects = objectnessSortIndices.sum()  

        # if there are no predicted objects AND no ground truth objects then we just skip this loop 
        if numPredictedObjects == 0 or numGroundTruthObjects == 0:
            continue
        
        # otherwise if both number of predicted objects and number of ground truth objects are both non-zero
        else:
            # find the cumulative sum of false positives 
            cumulativeFalsePositives = (1 - truePositives[objectnessSortIndices]).cumsum(0)
            # find the cumulative sum of true positives
            cumulativeTruePositives = truePositives[objectnessSortIndices].cumsum(0)

            # create the recall curve and append it to list
            recallCurve = cumulativeTruePositives / (numGroundTruthObjects + constVal)  
            # calculate recall at precisionScore
            recall[classIndex] = np.interp(-precisionScore, -objectnessVal[objectnessSortIndices], recallCurve[:, 0]) 

            # create the precision curve and append it to list
            precisionCurve = cumulativeTruePositives / (cumulativeTruePositives + cumulativeFalsePositives)  
            # calculate precision at precisionScore
            precision[classIndex] = np.interp(-precisionScore, -objectnessVal[objectnessSortIndices], precisionCurve[:, 0]) 

            # calculate AP from recall-precision curve
            for j in range(truePositives.shape[1]):
                AP[classIndex, j] = getAP(recallCurve[:, j], precisionCurve[:, j])

    # calculate F1 score
    F1 = 2 * precision * recall / (precision + recall + constVal)

    return precision, recall, AP, F1, uniqueClasses.astype('int32')


def getAP(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # append sentinel values at the beginning and end of the recall curve and precision curve
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))
    # calculate the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    # init a 101-point interp (COCO)
    x = np.linspace(0, 1, 101)
    # integrate area under envelope to calculate average precision
    AP = np.trapz(np.interp(x, mrec, mpre), x)

    return AP

# returns the IoU of box1 to box2. box1 is 4, box2 is nx4
def boundingBoxIOU(firstBox, secondBox, x1y1x2y2 = True, GIoU = False):
    
    # transpose secondBox
    secondBox = secondBox.t()
    # init const val 
    constVal = 1e-16

    if x1y1x2y2:
        # extract coordinates of bounding boxes - transform from center and width to exact coordinates
        firstBoxX1, firstBoxY1 = firstBox[0], firstBox[1]
        firstBoxX2, firstBoxY2 = firstBox[2], firstBox[3]
        secondBoxX1, secondBoxY1 = secondBox[0], secondBox[1]
        secondBoxX2, secondBoxY2 = secondBox[2], secondBox[3]

    else:  
        # extract coordinates of bounding boxes - transform from xywh to xyxy
        firstBoxX1, firstBoxX2 = firstBox[0] - firstBox[2] / 2, firstBox[0] + firstBox[2] / 2
        firstBoxY1, firstBoxY2 = firstBox[1] - firstBox[3] / 2, firstBox[1] + firstBox[3] / 2
        secondBoxX1, secondBoxX2 = secondBox[0] - secondBox[2] / 2, secondBox[0] + secondBox[2] / 2
        secondBoxY1, secondBoxY2 = secondBox[1] - secondBox[3] / 2, secondBox[1] + secondBox[3] / 2

    # extract intersection rectangle coordinates
    rectIntersectionX1, rectIntersectionY1  = torch.max(firstBoxX1, secondBoxX1), torch.max(firstBoxY1, secondBoxY1) 
    rectIntersectionX2, rectIntersectionY2 = torch.min(firstBoxX2, secondBoxX2), torch.min(firstBoxY2, secondBoxY2)
    
    # calculate intersection width
    intersectionWidth = (rectIntersectionX2 - rectIntersectionX1).clamp(0)
    # calculate intersection height
    intersectionHeight = (rectIntersectionY2 - rectIntersectionY1).clamp(0)
    # calculate intersection area 
    intersectionArea = intersectionWidth * intersectionHeight

    # calculate width and height of first box 
    firstWidth, firstHeight = firstBoxX2 - firstBoxX1, firstBoxY2 - firstBoxY1
    # calculate width and height of second box 
    secondWidth, secondHeight = secondBoxX2 - secondBoxX1, secondBoxY2 - secondBoxY1
    # calculate union area 
    unionArea = (firstWidth * firstHeight + constVal) + secondWidth * secondHeight - intersectionArea

    # calculate intersection-over-union (IoU) area
    iou = intersectionArea / unionArea  
    
    # check if GIoU is true 
    if GIoU:
        # extract smallest enclosing width (convex width)
        smallestEnclosingWidth = torch.max(firstBoxX2, secondBoxX2) - torch.min(firstBoxX1, secondBoxX1)  
        # extract smallest enclosing height (convex height)
        smallestEnclosingHeight = torch.max(firstBoxY2, secondBoxY2) - torch.min(firstBoxY1, secondBoxY1) 
        # calculate smallest enclosing area (convex araea) 
        smallestEnclosingArea = smallestEnclosingWidth * smallestEnclosingHeight + constVal 
        
        # return GIoU
        return iou - (smallestEnclosingArea - unionArea) / smallestEnclosingArea  

    return iou


def boxIOU(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    
    # calculate width and height of first box
    boxOneWidth = box1.t()[2] - box1.t()[0]
    boxOneHeight = box1.t()[3] - box1.t()[1]
    # calculate width and height of second box 
    boxTwoWdith = box2.t()[2] - box2.t()[0]
    boxTwoHeight = box2.t()[3] - box2.t()[1]
    # calculate area of first box
    areaOne = boxOneWidth * boxOneHeight
    # calculate area of second box 
    areaTwo = boxTwoWdith * boxTwoHeight

    # calculate intersection area 
    intersectionArea = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # calculate union area 
    unionArea = (areaOne[:, None] + areaTwo - intersectionArea)

    return intersectionArea /  unionArea

# returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
def widthHeightIOU(firstWidthHeight, secondWidthHeight):
    
    # extract shapes 
    firstWidthHeight = firstWidthHeight[:, None]  # [N,1,2]
    secondWidthHeight = secondWidthHeight[None]  # [1,M,2]
    # caclulate intersection area 
    intersectionArea = torch.min(firstWidthHeight, secondWidthHeight).prod(2)  # [N,M]
    # calculate union area 
    unionArea = (firstWidthHeight.prod(2) + secondWidthHeight.prod(2) - intersectionArea) 

    return intersectionArea / unionArea 

def getLosses(predictions, targets, model):  
    # init float tensor depending on cuda availability 
    FloatTensor = torch.cuda.FloatTensor if predictions[0].is_cuda else torch.Tensor

    # init class loss tensor to zeroes
    classLoss = FloatTensor([0])
    # init box loss tensor to zeroes 
    GIoUBoxLoss = FloatTensor([0])
    # init object loss tensor to zeroes 
    objectLoss = FloatTensor([0])

    # calculate and extract targets 
    tcls, tbox, indices, anchors = buildTargets(predictions, targets, model)  

    # define criteria for BCE loss
    BCEcls = nn.BCEWithLogitsLoss(pos_weight = FloatTensor([model.hyp['cls_pw']]), reduction = 'mean')
    BCEobj = nn.BCEWithLogitsLoss(pos_weight = FloatTensor([model.hyp['obj_pw']]), reduction = 'mean')

    # init total number of targets to zero 
    cumNumTargets = 0  
    
    # iterate through each layer predection (output )
    for layerIdx, layerPrediction in enumerate(predictions):
        # extract image index, anchor, y grid coordinate, x grid coordinate 
        imageIndex, anchor, gridY, gridX = indices[layerIdx]  
        # init target objectness value to tensor of zeroes 
        targetObj = torch.zeros_like(layerPrediction[..., 0])  
        # extract number of targets 
        numTargets = imageIndex.shape[0]  

        # check if number of targets is larger than zero 
        if numTargets:
            # increment cumulative number of targets with current number of targets 
            cumNumTargets += numTargets  
            # extract prediction subset corresponding to current targets
            predictionSubset = layerPrediction[imageIndex, anchor, gridY, gridX]  

            # extract prediction x, y coordinates 
            predictionXY = predictionSubset[:, :2].sigmoid()
            # extract prediction w,h values 
            predictionWH = predictionSubset[:, 2:4].exp().clamp(max = 1E3) * anchors[layerIdx]
            # create predicted boz by concatenating predictionXY and predictionWH
            predictedBox = torch.cat((predictionXY, predictionWH), 1) 
            # calculate GIoU
            GIoU = boundingBoxIOU(predictedBox.t(), tbox[layerIdx], x1y1x2y2 = False, GIoU = True) 
            # calculate GIoU box loss 
            GIoUBoxLoss += (1.0 - GIoU).mean()  

            # calculate objectness value (GIoU ratio)
            targetObj[imageIndex, anchor, gridY, gridX] = (1.0 - model.gr) + model.gr * GIoU.detach().clamp(0).type(targetObj.dtype)  

            # calculate and sum BCE class loss
            _targets = torch.full_like(predictionSubset[:, 5:], 0.0)  
            _targets[range(numTargets), tcls[layerIdx]] = 1.0
            classLoss += BCEcls(predictionSubset[:, 5:], _targets)  

        # calculate and sum object loss 
        objectLoss += BCEobj(layerPrediction[..., 4], targetObj) 

    # finalise values for GIoU box loss using hyperparameters
    GIoUBoxLoss *= model.hyp['giou']
    # finalise values for object loss using hyperparameters
    objectLoss *= model.hyp['obj']
    # finalise values for class loss using hyperparameters
    classLoss *= model.hyp['cls']

    # calculate total loss 
    totLoss = GIoUBoxLoss + objectLoss + classLoss

    return totLoss, torch.cat((GIoUBoxLoss, objectLoss, classLoss, totLoss)).detach()

# build targets for getLosses(), input targets(image,class,x,y,w,h)
def buildTargets(prediction, targets, model):
    
    # extract number of targets 
    numTargets = targets.shape[0]
    # init target classes, target boxes, target indices, target anchors to empty lists
    targetClasses, targetBoxes, targetIndices, targetAnchors = [], [], [], []
    # init gain to tensor filled with ones 
    gain = torch.ones(6, device = targets.device)

    # iterate through each layer in YOLO's layers
    for idx, layer in enumerate(model.yolo_layers):
        # extract anchors in current layer 
        anchors = model.module_list[layer].anchorVector
        # extract number of anchors
        numAnchors = anchors.shape[0]  
        # create anchor tensor 
        anchorTensor = torch.arange(numAnchors).view(numAnchors, 1).repeat(1, numTargets)  
        # calculate xyxy gain
        gain[2:] = torch.tensor(prediction[idx].shape)[[3, 2, 3, 2]] 

        # match targets to anchors
        
        # init layer anchor indices list 
        layerAnchorIndices = []
        # calculate scaled targets by multiplying by gain 
        scaledTargets = targets*gain
        # init offsets
        offsets = 0

        # check if number of targets is larger than zero 
        if numTargets:
            layer = widthHeightIOU(anchors, scaledTargets[:, 4:6]) > model.hyp['iou_t']  
            layerAnchorIndices, scaledTargets = anchorTensor[layer], scaledTargets.repeat(numAnchors, 1, 1)[layer]  #
            # overlaps
            gridXY = scaledTargets[:, 2:4]  

       # extract image index and image class
        imageIndex, imageClass = scaledTargets[:, :2].long().T 

        # gridX, gridY, gridW, gridH respectively represent the x, y, w, h on the grid
        # extract grid x,y values 
        gridXY = scaledTargets[:, 2:4]  
        # extract grid w,h values 
        gridWH = scaledTargets[:, 4:6]  

        # extract grid i,j values (grid x,y indices)
        # gridI, gridJ represent the integer part of x, y (which grid on the current feature map) - coords of upper left corner on feature map
        gridIJ = (gridXY - offsets).long()
        gridI, gridJ = gridIJ.T  

        # append values accordingly to corresponding lists 
        targetIndices.append((imageIndex, layerAnchorIndices, gridJ.clamp_(0, gain[3] - 1), gridI.clamp_(0, gain[2] - 1)))
        targetBoxes.append(torch.cat((gridXY - gridIJ, gridWH), 1))  
        targetAnchors.append(anchors[layerAnchorIndices]) 
        targetClasses.append(imageClass)  

    return targetClasses, targetBoxes, targetIndices, targetAnchors


def NMS(prediction, conf_thres = 0.1, iou_thres = 0.6, multi_label = True, classes = None, agnostic = False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # init minimum and maximum width and height 
    minBoxWH, maxBoxWH = 2, 4096  
    # extract number fo classes 
    numClasses = prediction[0].shape[1] - 5
    # multiple labels per box 
    multi_label &= numClasses > 1
    # init output list 
    output = [None] * prediction.shape[0]

    # iterate through images in prediction
    for imageIndex, imageInference in enumerate(prediction):  

        # apply confidence thresholding constraints and filter out the images that have a confidence score below our min threshold value
        imageInference = imageInference[imageInference[:, 4] > conf_thres]  
        # apply widht-height thresholding constraints and filter out the images that do not fall within the range min-max
        imageInference = imageInference[((imageInference[:, 2:4] > minBoxWH) & (imageInference[:, 2:4] < maxBoxWH)).all(1)]  # width-height

        # check if there are no detections remaining after filtering
        if not imageInference.shape[0]:
            continue

        # calculate confidence score by multiplying object confidence and class confidence together 
        imageInference[..., 5:] *= imageInference[..., 4:5]  

        # the bounding box attributes we have now are described by the center coordinates, as well as the height and width of the bounding box
        # however it is easier to calculate IoU of two boxes, using coordinates of a pair of diagnal corners for each box. 
        # so we want to  transform the (center x, center y, height, width) attributes of our boxes, to (top-left corner x, top-left corner y,  right-bottom corner x, right-bottom corner y) aka (x1,y1,x2,y2)
        box = xywh2xyxy(imageInference[:, :4])

        # create an Nx6 detection matrix (xyxy, conf, cls)
        nmsIndices, j = (imageInference[:, 5:] > conf_thres).nonzero().t()
        imageInference = torch.cat((box[nmsIndices], imageInference[nmsIndices, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)

        # check if classes is not none 
        if classes:
            # filter by classes
            imageInference = imageInference[(j.view(-1, 1) == torch.tensor(classes, device = j.device)).any(1)]

        # extract number of boxes
        numBoxes = imageInference.shape[0]  

        # check if there are no detections remaining after filtering
        if not numBoxes:
            continue

        # Batched NMS
        
        # extract number of classes 
        c = imageInference[:, 5] * 0 if agnostic else imageInference[:, 5]  
        # extract boxes offset by class and scores
        boxes, scores = imageInference[:, :4].clone() + c.view(-1, 1) * maxBoxWH, imageInference[:, 4]  
        # preform nms and store indices of elements to keep
        nmsIndices = torchvision.ops.boxes.nms(boxes, scores, iou_thres)

        # preform merge NMS using weighted mean 
        if (1 < numBoxes < 3E3):  
            try:  
                # create iou matrix 
                iou = boxIOU(boxes[nmsIndices], boxes) > iou_thres  # iou matrix
                # calculate box weights 
                weights = iou * scores[None]  
                # merge boxes 
                imageInference[nmsIndices, :4] = torch.mm(weights, imageInference[:, :4]).float() / weights.sum(1, keepdim = True)  
            except: 
                print(imageInference, nmsIndices, imageInference.shape, nmsIndices.shape)
                pass

        output[imageIndex] = imageInference[nmsIndices]

    return output

def convertToTarget(output, width, height):
    """
    Convert a YOLO model output to target format
    [batch_id, class_id, x, y, w, h, conf]
    """
    # check if output is a PyTorch tensor and convert to numpy array 
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    
    # init targets list 
    targets = []
    
    # iterate through outputs 
    for index, currOutput in enumerate(output):
        # check if current output is not empty 
        if currOutput is not None:
            # iterate through predictions in current output 
            for prediction in currOutput:
                # extract bounding box for current prediction
                box = prediction[:4]
                # extract width of bounding box 
                widthBox = (box[2] - box[0]) / width
                # extract height of bounding box 
                heightBox = (box[3] - box[1]) / height
                # extract x coordinate of bounding box
                xBox = box[0] / width + widthBox / 2
                # extract y coordinate of bounding box 
                yBox = box[1] / height + heightBox / 2
                # extract confidence score 
                conf = prediction[4]
                # extract box's predicted class 
                classID = int(prediction[5])
                # append to targets 
                targets.append([index, classID, xBox, yBox, widthBox, heightBox, conf])

    return np.array(targets)

def plotImages(images, targets, paths = None, fname ='images.jpg', names = None, max_size = 640, max_subplots = 16):
    
    # init line thickness 
    lineThickness = 3  
    #  init font thickness
    fontThickness = max(lineThickness - 1, 1)  
    
    # check if file arealdy exists and do not overrwrite 
    if os.path.isfile(fname):  
        return None
    # check if images are a PyTorch tensor and convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    # check if targets are a PyTorch tensor and convert to numpy 
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise images 
    if np.max(images[0]) <= 1:
        images *= 255

    # extract batchSize, height, width from image shape 
    batchSize, _, height, width = images.shape
    # calculate batch size as min of batch size and the max number of subplots   
    batchSize = min(batchSize, max_subplots)
    # calculate number of square subplots 
    numSubPlots = np.ceil(batchSize ** 0.5)  

    # calculate scale factor 
    scaleFactor = max_size / max(height, width)
    # check if resizing is necessary 
    if scaleFactor < 1:
        height = math.ceil(scaleFactor * height)
        width = math.ceil(scaleFactor * width)

    # init empty array for output
    mosaic = np.full((int(numSubPlots * height), int(numSubPlots * width), 3), 255, dtype = np.uint8)

    # craete class - colour lookup table 
    propertyCycle = plt.rcParams['axes.prop_cycle']
    hex2rgb = lambda height: tuple(int(height[1 + index:1 + index + 2], 16) for index in (0, 2, 4))
    colourLookUpTable = [hex2rgb(height) for height in propertyCycle.by_key()['color']]

    # iterate through images
    for index, img in enumerate(images):
        
        # check if we have reached max number of subplots 
        if index == max_subplots:  
            break
        
        # calculate block x value
        block_x = int(width * (index // numSubPlots))
        # calculate block y value 
        block_y = int(height * (index % numSubPlots))
        # transpose image accordingly 
        img = img.transpose(1, 2, 0)
        
        # check if image needs to be resized 
        if scaleFactor < 1:
            img = cv2.resize(img, (width, height))

        # assign image to mosaic 
        mosaic[block_y:block_y + height, block_x:block_x + width, :] = img
        
        # calculate number of targets 
        numTargets = len(targets) 

        # check if number of targets is larger than zero 
        if numTargets > 0:
            # extract image targets
            image_targets = targets[targets[:, 0] == index]
            # extract bounding boxes 
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            # extract classes 
            classes = image_targets[:, 1].astype('int')
            # ground truth if no confidence column
            groundTruth = image_targets.shape[1] == 6
            # check for confidence precense 
            conf = None if groundTruth else image_targets[:, 6]  

            boxes[[0, 2]] *= width
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= height
            boxes[[1, 3]] += block_y

            # iterate through boxes 
            for j, box in enumerate(boxes.T):
                # extract image class 
                imgCls = int(classes[j])
                imgCls = names[imgCls] if names else imgCls
                # extract colour from look-up table  
                color = colourLookUpTable[imgCls % len(colourLookUpTable)]
                
                # confidence threshold 
                if groundTruth or conf[j] > 0.3:
                    # extract label and plot box 
                    label = '%s' % imgCls if groundTruth else '%s %.1f' % (imgCls, conf[j])
                    plotBox(box, mosaic, label = label, color = color, line_thickness = lineThickness)

        # check if paths is not none and draw image filename labels
        if paths is not None:
            # trim label to fourty characters
            label = os.path.basename(paths[index])[:40] 
            # get text size 
            textSize = cv2.getTextSize(label, 0, fontScale = lineThickness / 3, thickness = fontThickness)[0]
            # add text to image 
            cv2.putText(mosaic, label, (block_x + 5, block_y + textSize[1] + 5), 0, lineThickness / 3, [220, 220, 220], thickness = fontThickness, lineType = cv2.LINE_AA)

        # create image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + width, block_y + height), (255, 255, 255), thickness = 3)

    # resize mosaic accordingly 
    mosaic = cv2.resize(mosaic, (int(numSubPlots * width * 0.5), int(numSubPlots * height * 0.5)), interpolation = cv2.INTER_AREA)
    # save mosaic  
    cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic

# plots one bounding box on image img
def plotBox(x, img, color = None, label = None, line_thickness = None):
    
    # init line thickness
    lineThickness = line_thickness 
    # start point and sned point for rectangle 
    startPoint, endPoint = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # draw rectangle on image 
    cv2.rectangle(img, startPoint, endPoint, color, thickness = lineThickness, lineType = cv2.LINE_AA)

    # check if label is not none 
    if label:
        # calculate font thickness 
        fontThickness = max(lineThickness - 1, 1)  
        # calculate text size 
        textSize = cv2.getTextSize(label, 0, fontScale = lineThickness / 3, thickness = fontThickness)[0]
        # cecalculate end point
        endPoint = startPoint[0] + textSize[0], startPoint[1] - textSize[1] - 3
        # draw rectangle for label and fill it 
        cv2.rectangle(img, startPoint, endPoint, color, -1, cv2.LINE_AA)  
        # place text in rectangle 
        cv2.putText(img, label, (startPoint[0], startPoint[1] - 2), 0, lineThickness / 3, [225, 255, 255], thickness = fontThickness, lineType = cv2.LINE_AA)

def plotResults(start = 0, stop = 0, bucket ='', id =()):  

    # create list of graph titles 
    graphTitles = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall', 'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']
    # create figure, axis instance 
    figure, axis = plt.subplots(2, 5, figsize =(12, 6), tight_layout = True)
    axis = axis.ravel()
    # extract files
    files = glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')
    
    # iterate through files
    for file in sorted(files):
        # load text from file and assign to results
        results = np.loadtxt(file, usecols =[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin = 2).T
        # extract number of rows
        numRows = results.shape[1] 
        x = numRows

        for i in range(10):
            y = results[i, x]

            # do not show loss values of zero 
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan
            
            # plot and set title 
            axis[i].plot(x, y, marker ='.', label = Path(file).stem, linewidth = 2, markersize = 8)
            axis[i].set_title(graphTitles[i])

    # show legend 
    axis[1].legend()
    # save figure as png
    figure.savefig('results.png', dpi = 200)
