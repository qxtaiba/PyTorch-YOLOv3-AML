import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

import test
from models import *
from utils.datasets import *
from utils.utils import *

# init output directories 
weightDi ory = 'weights' + os.sep
last = weightDirectory + 'last.pt'
best = weightDirectory + 'best.pt'
resultOutput = 'results.txt'

# init hyperparameters
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

    # extract path for cfg file
    configFilePath = opt.cfg

    # extract path for data file 
    dataFilePath = opt.data

    # extract number of epochs 
    numEpochs = opt.epochs

    # extract batch size 
    trainBatchSize = opt.batch_size

    # extract min image size, max image size, and test image size
    minImgSize, maxImgSize, testImgSize = opt.img_size  
    
    if minImgSize == maxImgSize:
        minImgSize //= 1.5
        maxImgSize //= 0.667

    # calculate accumulation interval
    accumulationInterval = max(round(64 / trainBatchSize), 1)

    # init grid size
    gridSize = 32  
    
    # extract min grid size and max grid size
    minGridSize, maxGridSize = minImgSize // gridSize, maxImgSize // gridSize

    # calculate min image size and max image size 
    minImgSize, maxImgSize = int(minGridSize * gridSize), int(maxGridSize * gridSize)

    # init image size with max image size 
    imgSize = maxImgSize

    # extract parsed data 
    parsedData = parse_data_cfg(dataFilePath)

    # extract training path
    trainingPath = parsedData['train']

    # extract testing path 
    testingPath = parsedData['valid']

    #extract number of classes
    numClasses =  int(parsedData['classes'])  

    # update coco-tuned hyp['cls'] to current dataset
    trainHyperParams['cls'] *= numClasses / 80  

    # attatch num classes to model
    model.nc = numClasses  

    # attatch hyperparametrs to model
    model.hyp = trainHyperParams  

    # attatch GIoU loss ratio to model 
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)

    beginningEpoch = 0
    bestFitnessScore = 0.0

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True


    # remove results from previous training sessions 
    for file in glob.glob('*_batch*.jpg') + glob.glob(resultOutput):
        os.remove(file)

    # init model
    model = Darknet(configFilePath).to(device)

    # init optimizer parameter groups
    paramGroupZero, paramGroupOne, paramGroupTwo = [], [], []  
    for key, value in dict(model.named_parameters()).items():
        if '.bias' in key:
            paramGroupTwo += [value]
        elif 'Conv2d.weight' in key:
            paramGroupOne += [value]  
        else:
            paramGroupZero += [value]  

    # set up optimizer 
    optimizer = optim.Adam(paramGroupZero, lr = trainHyperParams['lr0'])
    optimizer.add_param_group({'params': paramGroupOne, 'weight_decay': trainHyperParams['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': paramGroupTwo})  # add pg2 (biases)
    del paramGroupZero, paramGroupOne, paramGroupTwo

    # set up scheduler 
    lf = lambda x: (((1 + math.cos(x * math.pi / numEpochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lf)
    scheduler.last_epoch = beginningEpoch - 1 
    
    # initialise dataset
    dataset = LoadImagesAndLabels(trainingPath, imgSize, trainBatchSize, augment = True, hyp = trainHyperParams, cache_images = False)

    # initialise dataloader 
    trainBatchSize = min(trainBatchSize, len(dataset))
    numWorkers = min([os.cpu_count(), trainBatchSize if trainBatchSize > 1 else 0, 8])  # number of workers
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size = trainBatchSize, num_workers = numWorkers, shuffle = True,  pin_memory = True, collate_fn = dataset.collate_fn)

    # initialise dataloader to be used during testing 
    testDataLoader = torch.utils.data.DataLoader(LoadImagesAndLabels(testingPath, testImgSize, trainBatchSize, hyp = trainHyperParams, cache_images = False), batch_size = trainBatchSize, num_workers = numWorkers, pin_memory = True, collate_fn = dataset.collate_fn)

    # calculate num batches
    numBatches = len(dataLoader)  

    # calculate burn-in value 
    burnInVal = max(3 * numBatches, 500)  

    # init mAPs per class
    mAPs = np.zeros(numClasses)  

    # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    results = (0, 0, 0, 0, 0, 0, 0)  

    for epoch in range(beginningEpoch, numEpochs):
        
        # put model into training mode 
        model.train()

        # init mean loss values to zero 
        meanLoss = torch.zeros(4).to(device)  
        
        print(('\n' + '%10s' * 8) % ('Epoch', 'GPU_Mem', 'GIoU', 'Obj', 'Cls', 'Total', 'Targets', 'Img_Size'))
        
        # init progress bar 
        progressBar = tqdm(enumerate(dataLoader), total = numBatches)  

        for batchIdx, (imgs, targets, paths, null) in progressBar:

            # calculate number of completed batches
            numCompletedBatches = batchIdx + numBatches * epoch

            # convert from 0-255 to 0-1.0
            imgs = imgs.to(device).float() / 255.0 

            # send targets to device 
            targets = targets.to(device)

            # burn-in
            if numCompletedBatches <= burnInVal:
                xi = [0, burnInVal]  # x interp
                model.gr = np.interp(numCompletedBatches, xi, [0.0, 1.0])
                accumulationInterval = max(1, np.interp(numCompletedBatches, xi, [1, 64 / trainBatchSize]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(numCompletedBatches, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    x['weight_decay'] = np.interp(numCompletedBatches, xi, [0.0, trainHyperParams['weight_decay'] if j == 1 else 0.0])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(numCompletedBatches, xi, [0.9, trainHyperParams['momentum']])

            # multiscale
            if numCompletedBatches / accumulationInterval % 1 == 0:

                #  adjust img_size (67% - 150%) every 1 batch                
                imgSize = random.randrange(minGridSize, maxGridSize + 1) * gridSize

            # calculate scale factor 
            scaleFactor = imgSize / max(imgs.shape[2:]) 

            if scaleFactor != 1:
                newShape = [math.ceil(x * scaleFactor / gridSize) * gridSize for x in imgs.shape[2:]]  
                imgs = F.interpolate(imgs, size = newShape, mode ='bilinear', align_corners = False)

            # forward pass images through model 
            pred = model(imgs)

            # calculate losses 
            loss, lossItems = compute_loss(pred, targets, model)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', lossItems)
                return results

            # backwards pass
            loss *= trainBatchSize / 64
            loss.backward()

            # optimizer 
            if numCompletedBatches % accumulationInterval == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print training progress 
            meanLoss = (meanLoss * batchIdx + lossItems) / (batchIdx + 1)  
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, numEpochs - 1), mem, *meanLoss, len(targets), imgSize)
            progressBar.set_description(s)

            # plot results
            if numCompletedBatches < 1:
                f = 'train_batch%g.jpg' % batchIdx
                plot_images(images = imgs, targets = targets, paths = paths, fname = f)


        # update scheduler
        scheduler.step()

        # process epoch results
        isLastEpoch = epoch + 1 == numEpochs

        # calculate mAP
        if isLastEpoch:  
            results, mAPs = test.test(configFilePath, dataFilePath, batchSize = trainBatchSize, imgSize = testImgSize, model = model, dataloader = testDataLoader, multi_label = numCompletedBatches > burnInVal)

        # write results
        with open(resultOutput, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses =(GIoU, obj, cls)

        # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
        w = [0.0, 0.01, 0.99, 0.00]

        # update best mAP score
        arg = np.array(results).reshape(1, -1)
        fitnessScore = (arg[:, :4] * w).sum(1)  # fitness_i = weighted combination of [P, R, mAP, F1]

        if fitnessScore > bestFitnessScore:
            bestFitnessScore = fitnessScore

        # create checkpoint and save last/best model 
        if isLastEpoch:
            with open(resultOutput, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch, 'best_fitness': bestFitnessScore, 'training_results': f.read(), 'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict(), 'optimizer': None if isLastEpoch else optimizer.state_dict()}
            torch.save(ckpt, last)
            
            if (bestFitnessScore == fitnessScore) and not isLastEpoch:
                torch.save(ckpt, best)
            del ckpt


    plot_results()  # save as results.png
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type = int, default = 16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type = str, default ='cfg/yolov3.cfg', help ='*.cfg path')
    parser.add_argument('--data', type = str, default ='data/coco2017.data', help ='*.data path')
    parser.add_argument('--multi-scale', action ='store_true', help ='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs ='+', type = int, default =[320, 640], help ='[min_train, max-train, test]')
    parser.add_argument('--resume', action ='store_true', help ='resume training from last.pt')
    parser.add_argument('--name', default ='', help ='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default ='', help ='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # extend to 3 sizes (min, max, test)
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))

    train(trainHyperParams)  

