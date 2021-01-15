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

# Set printoptions
torch.set_printoptions(linewidth = 320, precision = 5, profile ='long')
np.set_printoptions(linewidth = 320, formatter ={'float_kind': '{:11.5g}'.format})  # format short g, %precision = 5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)

def xyxy2xywh(x):
    # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1 = top-left, xy2 = bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1 = top-left, xy2 = bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad = None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # Clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2

    return coords


def ap_per_class(tp, conf, pred_cls, target_cls):
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

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize =(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi = 300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(firstBox, secondBox, x1y1x2y2 = True, GIoU = False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    secondBox = secondBox.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        # Transform from center and width to exact coordinates
        firstBoxX1, firstBoxY1, firstBoxX2, firstBoxY2 = firstBox[0], firstBox[1], firstBox[2], firstBox[3]
        secondBoxX1, secondBoxY1, secondBoxX2, secondBoxY2 = secondBox[0], secondBox[1], secondBox[2], secondBox[3]
    else:  # transform from xywh to xyxy
        # Get the coordinates of bounding boxes
        firstBoxX1, firstBoxX2 = firstBox[0] - firstBox[2] / 2, firstBox[0] + firstBox[2] / 2
        firstBoxY1, firstBoxY2 = firstBox[1] - firstBox[3] / 2, firstBox[1] + firstBox[3] / 2
        secondBoxX1, secondBoxX2 = secondBox[0] - secondBox[2] / 2, secondBox[0] + secondBox[2] / 2
        secondBoxY1, secondBoxY2 = secondBox[1] - secondBox[3] / 2, secondBox[1] + secondBox[3] / 2

    # extract intersection rectangle coordinates
    rectIntersectionX1, rectIntersectionY1  = torch.max(firstBoxX1, secondBoxX1), torch.max(firstBoxY1, secondBoxY1) 
    rectIntersectionX2, rectIntersectionY2 = torch.min(firstBoxX2, secondBoxX2), torch.min(firstBoxY2, secondBoxY2)
    
    # Intersection area
    intersectionWidth = (rectIntersectionX2 - rectIntersectionX1).clamp(0)
    intersectionHeight = (rectIntersectionY2 - rectIntersectionY1).clamp(0)

    intersectionArea = intersectionWidth * intersectionHeight

    # Union Area
    firstWidth, firstHeight = firstBoxX2 - firstBoxX1, firstBoxY2 - firstBoxY1
    secondWidth, secondHeight = secondBoxX2 - secondBoxX1, secondBoxY2 - secondBoxY1
    unionArea = (firstWidth * firstHeight + 1e-16) + secondWidth * secondHeight - intersectionArea

    iou = intersectionArea / unionArea  # iou
    
    if GIoU:
        smallestEnclosingWidth = torch.max(firstBoxX2, secondBoxX2) - torch.min(firstBoxX1, secondBoxX1)  # convex (smallest enclosing box) width
        smallestEnclosingHeight = torch.max(firstBoxY2, secondBoxY2) - torch.min(firstBoxY1, secondBoxY1)  # convex height
        smallestEnclosingArea = smallestEnclosingWidth * smallestEnclosingHeight + 1e-16  # convex area
        return iou - (smallestEnclosingArea - unionArea) / smallestEnclosingArea  # GIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
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

    area1 = (box1.t()[2] - box1.t()[0]) * (box1.t()[3] - box1.t()[1])
    area2 = (box2.t()[2] - box2.t()[0]) * (box2.t()[3] - box2.t()[1])

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def compute_loss(p, targets, model):  # predictions, targets, model
    FloatTensor = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    classLoss, boxLoss, objectLoss = FloatTensor([0]), FloatTensor([0]), FloatTensor([0])
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight = FloatTensor([model.hyp['cls_pw']]), reduction = 'mean')
    BCEobj = nn.BCEWithLogitsLoss(pos_weight = FloatTensor([model.hyp['obj_pw']]), reduction = 'mean')

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = 1.0 - 0.5 * 0.0, 0.5 * 0.0

    # per output
    cumNumTargets = 0  # targets
    for layerIdx, layerPrediction in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[layerIdx]  # image, anchor, gridy, gridx
        targetObj = torch.zeros_like(layerPrediction[..., 0])  # target obj

        numTargets = b.shape[0]  # number of targets
        if numTargets:
            cumNumTargets += numTargets  # cumulative targets
            predictionSubset = layerPrediction[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = predictionSubset[:, :2].sigmoid()
            pwh = predictionSubset[:, 2:4].exp().clamp(max = 1E3) * anchors[layerIdx]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[layerIdx], x1y1x2y2 = False, GIoU = True)  # giou(prediction, target)
            boxLoss += (1.0 - giou).mean()  # giou loss

            # Obj
            targetObj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(targetObj.dtype)  # giou ratio

            # Class
            t = torch.full_like(predictionSubset[:, 5:], cn)  # targets
            t[range(numTargets), tcls[layerIdx]] = cp
            classLoss += BCEcls(predictionSubset[:, 5:], t)  # BCE

        objectLoss += BCEobj(layerPrediction[..., 4], targetObj)  # obj loss

    boxLoss *= model.hyp['giou']
    objectLoss *= model.hyp['obj']
    classLoss *= model.hyp['cls']

    totLoss = boxLoss + objectLoss + classLoss

    return totLoss, torch.cat((boxLoss, objectLoss, classLoss, totLoss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    numTargets = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device = targets.device)  # normalized to gridspace gain

    for idx, layer in enumerate(model.yolo_layers):
        anchors = model.module_list[layer].anchor_vec
        gain[2:] = torch.tensor(p[idx].shape)[[3, 2, 3, 2]]  # xyxy gain
        numAnchors = anchors.shape[0]  # number of anchors
        anchorTensor = torch.arange(numAnchors).view(numAnchors, 1).repeat(1, numTargets)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if numTargets:
            # r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            # j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
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

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending = True)]

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim = True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
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
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
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

def plot_results(start = 0, stop = 0, bucket ='', id =()):  # from utils.utils import *; plot_results()
    fig, ax = plt.subplots(2, 5, figsize =(12, 6), tight_layout = True)
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']

    files = glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')
    for f in sorted(files):
        results = np.loadtxt(f, usecols =[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin = 2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
                # y /= y[0]  # normalize
            ax[i].plot(x, y, marker ='.', label = Path(f).stem, linewidth = 2, markersize = 8)
            ax[i].set_title(s[i])
            # if i in [5, 6, 7]:  # share train and val loss y axes
            #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

    ax[1].legend()
    fig.savefig('results.png', dpi = 200)
