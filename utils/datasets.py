import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh, xywh2xyxy

acceptedImageFormats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

# find the orientation of the exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

# return the exif-corrected PIL size
def exif_size(img):

    #extract image size
    shape = img.size
    try:        
        # rotation by 270
        if dict(img._getexif().items())[orientation] == 6:
            shape = (shape[1], shape[0])
        # rotation by 90
        elif dict(img._getexif().items())[orientation] == 8:
            shape = (shape[1], shape[0])    
    except:
        pass

    return shape

def load_image(self, index):

    # extract image
    img = self.imgs[index]
    # extract path 
    path = self.img_files[index]
    # read image 
    img = cv2.imread(path)  
    # extract height and width of image 
    originalHeight, originalWidth = img.shape[:2]  
    # resize factor so that we can resize image to imageSize
    resizeFactor = self.imageSize / max(originalHeight, originalWidth)  # resize image to img_size
    
    # always resize down, only resize up if training with augmentation
    if resizeFactor != 1:
        # interpolate image
        interp = cv2.INTER_AREA if resizeFactor < 1 and not self.augment else cv2.INTER_LINEAR
        # resize image
        img = cv2.resize(img, (int(originalWidth * resizeFactor), int(originalHeight * resizeFactor)), interpolation = interp)
    
    # extract height and width of resized image 
    resizedHeight, resizedWidth = img.shape[:2]

    return img, (originalHeight, originalWidth), (resizedHeight, resizedWidth) 

def augment_hsv(img, hgain = 0.5, sgain = 0.5, vgain = 0.5):
    
    # init random gains
    randomGains = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  
    # extract hue, saturation, value from image
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # init numpy array
    x = np.arange(0, 256, dtype = np.int16)
    # init look-up table for hue with random gain
    lookUpHue = ((x * randomGains[0]) % 180).astype(img.dtype)
    # init look-up table for saturation with random gain 
    lookUpSat = np.clip(x * randomGains[1], 0, 255).astype(img.dtype)
    # init look-up table for value with random gain
    lookUpVal = np.clip(x * randomGains[2], 0, 255).astype(img.dtype)
    # extract new hue, saturation, value for image using look-up tables
    modifiedHSV = cv2.merge((cv2.LUT(hue, lookUpHue), cv2.LUT(sat, lookUpSat), cv2.LUT(val, lookUpVal))).astype(img.dtype)
    # modify image
    cv2.cvtColor(modifiedHSV, cv2.COLOR_HSV2BGR, dst = img)  

def mosaic(self, index):   

    # init labels 
    mosaicLabels = []
    # extract image size
    imageSize = self.imageSize
    # randomly init center coordinates
    centerX, centerY = [int(random.uniform(imageSize * 0.5, imageSize * 1.5)) for _ in range(2)]
    # randomly init an additional three image indices
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]
    
    for i, imageIndex in enumerate(indices):
        # load current image
        img, (originalHeight, originalWidth), (resizedHeight, resizedWidth) = load_image(self, imageIndex)

        if i == 0: # top left
            # create base image with 4 tiles
            baseImage = np.full((imageSize * 2, imageSize * 2, img.shape[2]), 114, dtype = np.uint8)  
            # xmin, ymin, xmax, ymax for large image
            xMinlarge, yminLarge, xMaxLarge, yMaxLarge = max(centerX - resizedWidth, 0), max(centerY - resizedHeight, 0), centerX, centerY
            # xmin, ymin, xmax, ymax for small image
            xMinSmall, yMinSmall, xMaxSmall, yMaxSmall = resizedWidth - (xMaxLarge - xMinlarge), resizedHeight - (yMaxLarge - yminLarge), resizedWidth, resizedHeight  
        
        elif i == 1:  # top right
            # xmin, ymin, xmax, ymax for large image
            xMinlarge, yminLarge, xMaxLarge, yMaxLarge = centerX, max(centerY - resizedHeight, 0), min(centerX + resizedWidth, imageSize * 2), centerY
            # xmin, ymin, xmax, ymax for small image
            xMinSmall, yMinSmall, xMaxSmall, yMaxSmall = 0, resizedHeight - (yMaxLarge - yminLarge), min(resizedWidth, xMaxLarge - xMinlarge), resizedHeight
        
        elif i == 2: # bottom left
            # xmin, ymin, xmax, ymax for large image
            xMinlarge, yminLarge, xMaxLarge, yMaxLarge = max(centerX - resizedWidth, 0), centerY, centerX, min(imageSize * 2, centerY + resizedHeight)
            # xmin, ymin, xmax, ymax for small image
            xMinSmall, yMinSmall, xMaxSmall, yMaxSmall = resizedWidth - (xMaxLarge - xMinlarge), 0, max(centerX, resizedWidth), min(yMaxLarge - yminLarge, resizedHeight)
        
        elif i == 3: # bottom right
            # xmin, ymin, xmax, ymax for large image
            xMinlarge, yminLarge, xMaxLarge, yMaxLarge = centerX, centerY, min(centerX + resizedWidth, imageSize * 2), min(imageSize * 2, centerY + resizedHeight)
            # xmin, ymin, xmax, ymax for small image
            xMinSmall, yMinSmall, xMaxSmall, yMaxSmall = 0, 0, min(resizedWidth, xMaxLarge - xMinlarge), min(yMaxLarge - yminLarge, resizedHeight)

        # init base image parameters 
        baseImage[yminLarge:yMaxLarge, xMinlarge:xMaxLarge] = img[yMinSmall:yMaxSmall, xMinSmall:xMaxSmall]  
        
        # calculate padding 
        widthPadding = xMinlarge - xMinSmall
        heightPadding = yminLarge - yMinSmall

        # extract labels
        labels = self.labels[imageIndex]
        _labels = labels.copy()

        # normalize xywh to xyxy format
        if labels.size > 0:
            _labels[:, 1] = resizedWidth * (labels[:, 1] - labels[:, 3] / 2) + widthPadding
            _labels[:, 2] = resizedHeight * (labels[:, 2] - labels[:, 4] / 2) + heightPadding
            _labels[:, 3] = resizedWidth * (labels[:, 1] + labels[:, 3] / 2) + widthPadding
            _labels[:, 4] = resizedHeight * (labels[:, 2] + labels[:, 4] / 2) + heightPadding
        
        mosaicLabels.append(_labels)

    # check if mosaicLabels is not empty
    if len(mosaicLabels):
        # concatenate labels
        mosaicLabels = np.concatenate(mosaicLabels, 0)
        # clip labels
        np.clip(mosaicLabels[:, 1:], 0, 2 * imageSize, out = mosaicLabels[:, 1:])

    # augment images and labels
    baseImage, mosaicLabels = randAffine(baseImage, mosaicLabels,degrees = self.hyp['degrees'], translate = self.hyp['translate'], scale = self.hyp['scale'], shear = self.hyp['shear'], border = -imageSize // 2)  # border to remove

    return baseImage, mosaicLabels


def letterbox(img, newShape = (416, 416), color = (114, 114, 114), auto = True, scaleFill = False, scaleup = True):

    # extract current image shape
    currShape = img.shape[:2]

    # check if new image shape is an integer or a tuple
    if isinstance(newShape, int):
        # create tuple
        newShape = (newShape, newShape)
    
    # calculate scale ratio by dividing new shape by old shape
    scaleRatio = min(newShape[0] / currShape[0], newShape[1] / currShape[1])

    # only scale down, do not scale up
    if not scaleup:
        scaleRatio = min(scaleRatio, 1.0)

    # extract unpadded shape
    unpaddedShape = (int(round(currShape[1] * scaleRatio)), int(round(currShape[0] * scaleRatio)))
    # calculate width and height padding 
    widthPadding, heightPadding = newShape[1] - unpaddedShape[0], newShape[0] - unpaddedShape[1]  
    
    if auto:  
        widthPadding, heightPadding = np.mod(widthPadding, 32), np.mod(heightPadding, 32)  # wh padding

    # resize image if current shape does not equal the unpadded shape 
    if currShape[::-1] != unpaddedShape:
        img = cv2.resize(img, unpaddedShape, interpolation = cv2.INTER_LINEAR)

    # divide width padding into two sides (left/right)
    widthPadding /= 2
    # divide height padding into two side (bottom/above)
    heightPadding /= 2
    # create top/bottom border
    top, bottom = int(round(heightPadding - 0.1)), int(round(heightPadding + 0.1))
    # create left/right border 
    left, right = int(round(widthPadding - 0.1)), int(round(widthPadding + 0.1))
    # add borders
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)  

    return img, (scaleRatio, scaleRatio), (widthPadding, heightPadding)


def randAffine(img, targets =(), degrees = 10, translate =.1, scale =.1, shear = 10, border = 0):

    # calculate height
    height = img.shape[0] + border * 2
    # calculate width 
    width = img.shape[1] + border * 2

    # rotate and scale by first creating 3x3 identity matrix 
    R = np.eye(3)
    # init random angle value 
    angle = random.uniform(-degrees, degrees)
    # init random scale valye 
    scale = random.uniform(1 - scale, 1 + scale)
    # create rotation matrix
    R[:2] = cv2.getRotationMatrix2D(angle = angle, center =(img.shape[1] / 2, img.shape[0] / 2), scale = scale)

    # translate by first creating 3x3 identity matrix
    T = np.eye(3)
    # init x translation
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  
    # init y translation
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  

    # shear by first creating 3x3 identity matrix
    S = np.eye(3)
    # init x shear [deg]
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  
    # init y shear [deg] 
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  

    # creatte combined rotation matrix
    M = S @ T @ R

    # check if image changed
    if (border != 0) or (M != np.eye(3)).any():
        img = cv2.warpAffine(img, M[:2], dsize = (width, height), flags = cv2.INTER_LINEAR, borderValue = (114, 114, 114))

    # transform label coordinates
    if len(targets):

        # warp points
        xy = np.ones((len(targets) * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(len(targets) * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(len(targets), 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, len(targets)).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        
        # extract width
        w = xy[:, 2] - xy[:, 0]
        # extract height
        h = xy[:, 3] - xy[:, 1]

        # calculate area 
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        
        # calculate aspect ratio
        aspectRatio = np.maximum(w/(h + 1e-16), h/(w + 1e-16))  
        
        i = (w > 4) & (h > 4) & (area / (area0 * scale + 1e-16) > 0.2) & (aspectRatio < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


class LoadImages: 
    def __init__(self, path, img_size = 416):
        
        # init files list  
        files = []
        # extract path 
        path = str(Path(path))  
        
        # check if path leads to a directory  and populate files list
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        # check if path leads to a file and populate files list 
        elif os.path.isfile(path):
            files = [path]

        # extract image(s) if they are in the correct format 
        images = [x for x in files if os.path.splitext(x)[-1].lower() in acceptedImageFormats]
        # extract number of images
        numImages = len(images)
        # init image size 
        self.imgSize = img_size
        # inite files 
        self.files = images 
        # init number of files 
        self.numFiles = numImages 

    def __iter__(self):

        # init count to zero 
        self.count = 0

        return self

    def __next__(self):

        # check if we have loaded all of the images 
        if self.count == self.numFiles:
            raise StopIteration
        
        # extract path
        path = self.files[self.count]
        # increment count
        self.count += 1
        # read image
        img0 = cv2.imread(path)
        # resize by adding padding
        img = letterbox(img0, newShape = self.imgSize)[0]
        # convert image from BGR to RGB and to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return path, img, img0

    def __len__(self):

        # return number of files
        return self.numFiles

class LoadImagesAndLabels(Dataset):  
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, cache_images=False, single_cls=False, pad=0.0):
        
        # extract path 
        path = str(Path(path))  
        # extract parent 
        parent = str(Path(path).parent) + os.sep

        # read through file and adjust lines 
        with open(path, 'r') as f:
            f = f.read().splitlines()
            f = [x.replace('./', parent) if x.startswith('./') else x for x in f] 

        # extract image files if they are in the correct format 
        self.imgFiles = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in acceptedImageFormats]
        # extract label files
        self.labelFiles = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in self.imgFiles]

        # extract number of files
        numFiles = len(self.imgFiles)
        # extract batch index
        imageBatchIndex = np.floor(np.arange(numFiles) / batch_size).astype(np.int)  

        # init number of images 
        self.numImages = numFiles  
        # init image batch index 
        self.imageBatchIndex = imageBatchIndex  
        # init image size 
        self.imageSize = img_size
        # init augment bool
        self.isAugment = augment
        # init hyperparameters
        self.hyperparameters = hyp
        # init mosaic bool
        self.isMosaic = self.isAugment   
        
        # extract image shapes 
        shapefile = [exif_size(Image.open(f)) for f in self.imgFiles]
        # init shapes array 
        self.shapes = np.array(shapefile, dtype=np.float64)

        # init image list 
        self.imgs = [None] * numFiles
        # init labels list
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * numFiles
        # init bool to check if labels are already cached
        isLabelsLoaded = False
        # init numMissing, numFound, numDuplicate
        numMissing, numFound, numEmpty, numDuplicate = 0, 0, 0, 0  
        # init path to save/retrieve cached labels in .npy file
        cachedLabelsFile = str(Path(self.labelFiles[0]).parent) + '.npy' 

        # check if cached labels file exists
        if os.path.isfile(cachedLabelsFile):
            shapefile = cachedLabelsFile
            # load cached labels file
            x = np.load(cachedLabelsFile, allow_pickle=True)
            # check if number of cached labels is equal to total number of files/labels
            if len(x) == numFiles:
                # assign labels
                self.labels = x
                # set isLabelsLoaded to true 
                isLabelsLoaded = True
        else:
            # replace all instances of 'images' with 'labels' 
            shapefile = path.replace('images', 'labels')

        # create a progress bar and iterate through label files
        progressBar = tqdm(self.labelFiles)
        for index, file in enumerate(progressBar):

            # check if label is already loaded 
            if isLabelsLoaded:
                # assign current indexed label
                label = self.labels[index]

            else:
                try:
                    # open file
                    with open(file, 'r') as f:
                        # parse through file and assign to label
                        label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    # if except is triggered then this means we have a missing file/label and increment our count 
                    numMissing += 1  
                    continue
            
            # check if label is empty
            if label.shape[0]:
                # check if any duplicate rows
                if np.unique(label, axis=0).shape[0] < label.shape[0]:  
                    # if there are duplcicate rows then we have a duplicate file/label and increment our count 
                    numDuplicate += 1  

                # assing extracted label calue 
                self.labels[index] = label
                # increment our count of number of files found 
                numFound += 1  

            else:
                # increment our count of empty files
                numEmpty += 1

            # add descriptive values to progress bar 
            progressBar.desc = 'caching labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (shapefile, numFound, numMissing, numEmpty, numDuplicate, numFiles)

        # if labels not cached already, save for next training
        if not isLabelsLoaded and numFiles > 1000:
            np.save(cachedLabelsFile, self.labels) 


    def __len__(self):
        return len(self.imgFiles)

    def __getitem__(self, index):
        
        #extract hyperparameters 
        hyp = self.hyperparameters

        # check if isMosaic is true
        if self.isMosaic:
            # create mosaic and extract the tiled base image and corresponding labels
            img, labels = mosaic(self, index)
            shapes = None

        else:
            # load image 
            img, (originalHeight, originalWeight), (resizedHeight, resizedWidth) = load_image(self, index)
            # extract image size 
            shape = self.imgSize
            # letterbox image and extract the letterboxed image, the scale ratio used, and padding
            img, ratio, pad = letterbox(img, shape, auto = False, scaleup = self.isAugment)
            # COCO mAP rescaling 
            shapes = (originalHeight, originalWeight), ((resizedHeight / originalHeight, resizedWidth / originalWeight), pad)  

            # init labels list
            labels = []
            # extract labels 
            x = self.labels[index]

            if x.size > 0:
                # make a copy of extracted labels 
                labels = x.copy()
                # normalize xywh to xyxy format
                labels[:, 1] = ratio[0] * resizedWidth * (x[:, 1] - x[:, 3] / 2) + pad[0]  
                labels[:, 2] = ratio[1] * resizedHeight * (x[:, 2] - x[:, 4] / 2) + pad[1]  
                labels[:, 3] = ratio[0] * resizedWidth * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * resizedHeight * (x[:, 2] + x[:, 4] / 2) + pad[1]

        # check if isAugment is true 
        if self.isAugment:
            # check if isMosaic is true 
            if not self.isMosaic:
                img, labels = randAffine(img, labels, degrees = hyp['degrees'], translate = hyp['translate'], scale = hyp['scale'], shear = hyp['shear'])
            # augment image/color space
            augment_hsv(img, hgain = hyp['hsv_h'], sgain = hyp['hsv_s'], vgain = hyp['hsv_v'])
        
        # check if labels is not empty 
        if len(labels) :
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            # normalize height 
            labels[:, [2, 4]] /= img.shape[0]  
            # normalize width 
            labels[:, [1, 3]] /= img.shape[1]  

        # check if isAugment is true 
        if self.isAugment:
            # randomly preform left/right flip  on image
            if random.random() < 0.5:
                img = np.fliplr(img)
                # flip labels 
                if len(labels) :
                    labels[:, 1] = 1 - labels[:, 1]

        # init output labels 
        outputLabels = torch.zeros((len(labels) , 6))
        # check if labels is not empty 
        if len(labels):
            # populate output labels 
            outputLabels[:, 1:] = torch.from_numpy(labels)

        # convert from BGR to RGB and reshape to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), outputLabels, self.imgFiles[index], shapes

    @staticmethod
    def collate_fn(batch):
        # extract image, label, path, shapes 
        img, label, path, shapes = zip(*batch)  

        # iterate through labels 
        for i, l in enumerate(label):
            # prepend target image index 
            l[:, 0] = i

        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

