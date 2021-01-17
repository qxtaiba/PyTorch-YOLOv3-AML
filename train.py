import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

weightDirectory = 'weights' + os.sep  # weights dir
last = weightDirectory + 'last.pt'
best = weightDirectory + 'best.pt'
resultOutput = 'results.txt'

# Hyperparameters
trainHyperParams = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*= img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD = 5E-3, Adam = 5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma = 1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

def train(trainHyperParams):

    configFilePath = opt.cfg
    dataFilePath = opt.data
    numEpochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    trainBatchSize = opt.batch_size
    accumulationInterval = max(round(64 / trainBatchSize), 1)  # accumulate n times before optimizer update (bs 64)
    minImgSize, maxImgSize, testImgSize = opt.imageSize  # img sizes (min, max, test)

    # Image Sizes
    gridSize = 32  # (pixels) grid size
    
    if minImgSize == maxImgSize:
        minImgSize //= 1.5
        maxImgSize //= 0.667

    minGridSize, maxGridSize = minImgSize // gridSize, maxImgSize // gridSize
    minImgSize, maxImgSize = int(minGridSize * gridSize), int(maxGridSize * gridSize)
    imgSize = maxImgSize  # initialize with max size

    # Configure run
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Reduce randomness 
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True

    parsedData = parseData(dataFilePath)
    trainingPath = parsedData['train']
    testingPath = parsedData['valid']
    numClasses =  int(parsedData['classes'])  # number of classes
    trainHyperParams['cls'] *= numClasses / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for file in glob.glob('*_batch*.jpg') + glob.glob(resultOutput):
        os.remove(file)

    # Initialize model
    model = Darknet(configFilePath).to(device)

    # Optimizer
    paramGroupZero, paramGroupOne, paramGroupTwo = [], [], []  # optimizer parameter groups
    for key, value in dict(model.named_parameters()).items():
        if '.bias' in key:
            paramGroupTwo += [value]  # biases
        elif 'Conv2d.weight' in key:
            paramGroupOne += [value]  # apply weight_decay
        else:
            paramGroupZero += [value]  # all else

    optimizer = optim.Adam(paramGroupZero, lr = trainHyperParams['lr0'])
    optimizer.add_param_group({'params': paramGroupOne, 'weight_decay': trainHyperParams['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': paramGroupTwo})  # add pg2 (biases)
    del paramGroupZero, paramGroupOne, paramGroupTwo

    beginningEpoch = 0
    bestFitnessScore = 0.0
    
    lf = lambda x: (((1 + math.cos(x * math.pi / numEpochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lf)
    scheduler.last_epoch = beginningEpoch - 1 
    
    # Dataset
    print(trainingPath)
    dataset = LoadImagesAndLabels(trainingPath, imgSize, trainBatchSize, augment = True, hyp = trainHyperParams, rect = False, cache_images = False)

    # Dataloader
    trainBatchSize = min(trainBatchSize, len(dataset))
    numWorkers = min([os.cpu_count(), trainBatchSize if trainBatchSize > 1 else 0, 8])  # number of workers
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size = trainBatchSize, num_workers = numWorkers, shuffle = True,  pin_memory = True, collate_fn = dataset.collate_fn)

    # Testloader
    testDataLoader = torch.utils.data.DataLoader(LoadImagesAndLabels(testingPath, testImgSize, trainBatchSize, hyp = trainHyperParams, rect = True, cache_images = False), batch_size = trainBatchSize, num_workers = numWorkers, pin_memory = True, collate_fn = dataset.collate_fn)

    # Model parameters
    model.nc = numClasses  # attach number of classes to model
    model.hyp = trainHyperParams  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)

    # Start training
    numBatches = len(dataLoader)  # number of batches
    burnInVal = max(3 * numBatches, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    mAPs = np.zeros(numClasses)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    beginningTime = time.time()

    for epoch in range(beginningEpoch, numEpochs):
        model.train()

        meanLoss = torch.zeros(4).to(device)  # mean losses
        
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        
        progressBar = tqdm(enumerate(dataLoader), total = numBatches)  # progress bar

        for batchIdx, (imgs, targets, paths, null) in progressBar:
            numCompletedBatches = batchIdx + numBatches * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            if numCompletedBatches <= burnInVal:
                xi = [0, burnInVal]  # x interp
                model.gr = np.interp(numCompletedBatches, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulationInterval = max(1, np.interp(numCompletedBatches, xi, [1, 64 / trainBatchSize]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(numCompletedBatches, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    x['weight_decay'] = np.interp(numCompletedBatches, xi, [0.0, trainHyperParams['weight_decay'] if j == 1 else 0.0])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(numCompletedBatches, xi, [0.9, trainHyperParams['momentum']])

            # Multi-Scale
            if numCompletedBatches / accumulationInterval % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                imgSize = random.randrange(minGridSize, maxGridSize + 1) * gridSize
            scaleFactor = imgSize / max(imgs.shape[2:])  # scale factor
            if scaleFactor != 1:
                newShape = [math.ceil(x * scaleFactor / gridSize) * gridSize for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                imgs = F.interpolate(imgs, size = newShape, mode ='bilinear', align_corners = False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, lossItems = getLosses(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', lossItems)
                return results

            # Backward
            loss *= trainBatchSize / 64  # scale loss
            loss.backward()

            # Optimize
            if numCompletedBatches % accumulationInterval == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print
            meanLoss = (meanLoss * batchIdx + lossItems) / (batchIdx + 1)  # update mean losses
            
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, numEpochs - 1), mem, *meanLoss, len(targets), imgSize)
            progressBar.set_description(s)

            # Plot
            if numCompletedBatches < 1:
                f = 'train_batch%g.jpg' % batchIdx  # filename
                res = plotImages(images = imgs, targets = targets, paths = paths, fname = f)

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        isLastEpoch = epoch + 1 == numEpochs
        if isLastEpoch:  # Calculate mAP
            is_coco = any([x in dataFilePath for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, mAPs = test.test(configFilePath, dataFilePath, batchSize = trainBatchSize, imgSize = testImgSize, model = model, dataloader = testDataLoader, multi_label = numCompletedBatches > burnInVal)

        # Write
        with open(resultOutput, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses =(GIoU, obj, cls)


        # Update best mAP
        arg = np.array(results).reshape(1, -1)
        w = [0.0, 0.01, 0.99, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
        fitnessScore = (arg[:, :4] * w).sum(1)  # fitness_i = weighted combination of [P, R, mAP, F1]

         
        if fitnessScore > bestFitnessScore:
            bestFitnessScore = fitnessScore

        # Save model
        if isLastEpoch:
            with open(resultOutput, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch, 'best_fitness': bestFitnessScore, 'training_results': f.read(), 'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict(), 'optimizer': None if isLastEpoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if (bestFitnessScore == fitnessScore) and not isLastEpoch:
                torch.save(ckpt, best)
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    plotResults()  # save as results.png
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type = int, default = 16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type = str, default ='cfg/yolov3.cfg', help ='*.cfg path')
    parser.add_argument('--data', type = str, default ='data/coco2017.data', help ='*.data path')
    parser.add_argument('--multi-scale', action ='store_true', help ='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--imageSize', nargs ='+', type = int, default =[320, 640], help ='[min_train, max-train, test]')
    parser.add_argument('--resume', action ='store_true', help ='resume training from last.pt')
    parser.add_argument('--name', default ='', help ='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default ='', help ='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    
    print(opt)
    opt.imageSize.extend([opt.imageSize[-1]] * (3 - len(opt.imageSize)))  # extend to 3 sizes (min, max, test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(trainHyperParams)  # train normally
