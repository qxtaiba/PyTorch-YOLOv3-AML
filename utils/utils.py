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
def scale_coords(img1_shape, coords, img0_shape, ratio_pad = None):
    
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
def ap_per_class(truePositives, objectnessVal, predictedClasses, targetClasses):
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
                AP[classIndex, j] = compute_ap(recallCurve[:, j], precisionCurve[:, j])

    # calculate F1 score
    F1 = 2 * precision * recall / (precision + recall + constVal)

    return precision, recall, AP, F1, uniqueClasses.astype('int32')


def compute_ap(recall, precision):
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
def bbox_iou(firstBox, secondBox, x1y1x2y2 = True, GIoU = False):
    
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


def box_iou(box1, box2):
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
def wh_iou(firstWidthHeight, secondWidthHeight):
    
    # extract shapes 
    firstWidthHeight = firstWidthHeight[:, None]  # [N,1,2]
    secondWidthHeight = secondWidthHeight[None]  # [1,M,2]
    # caclulate intersection area 
    intersectionArea = torch.min(firstWidthHeight, secondWidthHeight).prod(2)  # [N,M]
    # calculate union area 
    unionArea = (firstWidthHeight.prod(2) + secondWidthHeight.prod(2) - intersectionArea) 

    return intersectionArea / unionArea 

def compute_loss(predictions, targets, model):  
    # init float tensor depending on cuda availability 
    FloatTensor = torch.cuda.FloatTensor if predictions[0].is_cuda else torch.Tensor

    # init class loss tensor to zeroes
    classLoss = FloatTensor([0])
    # init box loss tensor to zeroes 
    GIoUBoxLoss = FloatTensor([0])
    # init object loss tensor to zeroes 
    objectLoss = FloatTensor([0])

    # calculate and extract targets 
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  

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
            GIoU = bbox_iou(predictedBox.t(), tbox[layerIdx], x1y1x2y2 = False, GIoU = True) 
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


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    numTargets = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device = targets.device)  # normalized to gridspace gain

    for idx, layer in enumerate(model.yolo_layers):
        anchors = model.module_list[layer].anchorVector
        gain[2:] = torch.tensor(p[idx].shape)[[3, 2, 3, 2]]  # xyxy gain
        numAnchors = anchors.shape[0]  # number of anchors
        anchorTensor = torch.arange(numAnchors).view(numAnchors, 1).repeat(1, numTargets)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if numTargets:

            layer = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = anchorTensor[layer], t.repeat(numAnchors, 1, 1)[layer]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy

       # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch


def non_max_suppression(prediction, conf_thres = 0.1, iou_thres = 0.6, multi_label = True, classes = None, agnostic = False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = True  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        i, j = (x[:, 5:] > conf_thres).nonzero().t()
        x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device = j.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim = True)  # merged boxes
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def output_to_target(output, width, height):
    """
    Convert a YOLO model output to target format
    [batch_id, class_id, x, y, w, h, conf]
    """
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)


def plot_one_box(x, img, color = None, label = None, line_thickness = None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness = tl, lineType = cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale = tl / 3, thickness = tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness = tf, lineType = cv2.LINE_AA)


def plot_images(images, targets, paths = None, fname ='images.jpg', names = None, max_size = 640, max_subplots = 16):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if os.path.isfile(fname):  # do not overwrite
        return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype = np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label = label, color = color, line_thickness = tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale = tl / 3, thickness = tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness = tf,
                        lineType = cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness = 3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation = cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic

def plot_results(start = 0, stop = 0, bucket ='', id =()):  
    fig, ax = plt.subplots(2, 5, figsize =(12, 6), tight_layout = True)
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall', 'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']

    files = glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')
    for f in sorted(files):
        results = np.loadtxt(f, usecols =[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin = 2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
            ax[i].plot(x, y, marker ='.', label = Path(f).stem, linewidth = 2, markersize = 8)
            ax[i].set_title(s[i])

    ax[1].legend()
    fig.savefig('results.png', dpi = 200)
