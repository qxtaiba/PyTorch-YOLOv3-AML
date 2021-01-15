import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(cfg,data, weights=None, batchSize=16, imgSize=416, confidenceThreshold=0.001, iouThreshold=0.6, augment=False, model=None, dataloader=None, multi_label=True):
    
    # Initialize/load model and set device
    if model is None:
        isTraining = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        verbose = True

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, imgSize)

        # Load weights
        model.load_state_dict(torch.load(weights, map_location=device)['model'])

        # Fuse
        model.fuse()
        model.to(device)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        isTraining = True
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    numClasses =  int(data['classes'])  # number of classes
    testPath = data['valid']  # path to test images
    with open(data['names'], 'r') as f:
        names = f.read().split('\n')
    classNames = list(filter(None, names))  # filter removes empty strings (such as last line) # class names

    
    iouVector = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouVector = iouVector[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouVector.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(testPath, imgSize, batchSize, rect=True, pad=0.5)
        batchSize = min(batchSize, len(dataset))
        dataloader = DataLoader(dataset, batchSize=batchSize, num_workers=min([os.cpu_count(), batchSize if batchSize > 1 else 0, 8]), pin_memory=True, collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    _ = model(torch.zeros((1, 3, imgSize, imgSize), device=device)) if device.type != 'cpu' else None  # run once
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    precision, recall, F1, meanPrecision, meanRecall, mAP, meanF1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    testStatistics, AP, APClass = [], [], []

    for batchIdx, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        numBatches, channels, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            inferenceOutput, trainingOutput = model(imgs, augment=augment)  # inference and training outputs

            # Compute loss
            if isTraining:  # if model has loss hyperparameters
                loss += compute_loss(trainingOutput, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            output = non_max_suppression(inferenceOutput, conf_thres=confidenceThreshold, iou_thres=iouThreshold, multi_label=multi_label)

        # Statistics per image
        for statIdx, pred in enumerate(output):
            labels = targets[targets[:, 0] == statIdx, 1:]
            numLabels = len(labels)
            targetClass = labels[:, 0].tolist() if numLabels else []  # target class
            seen += 1

            if pred is None:
                if numLabels:
                    testStatistics.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), targetClass))
                continue


            # Clip boxes to image bounds
            # Clip bounding xyxy bounding boxes to image shape (height, width)
            pred[:, 0].clamp_(0, width)  # x1
            pred[:, 1].clamp_(0, height)  # y1
            pred[:, 2].clamp_(0, width)  # x2
            pred[:, 3].clamp_(0, height)  # y2

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if numLabels:
                detected = []  # target indices
                targetClassTensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(targetClassTensor):
                    targetIndices = (cls == targetClassTensor).nonzero().view(-1) 
                    predictionIndices = (cls == pred[:, 5]).nonzero().view(-1)  

                    # Search for detections
                    if predictionIndices.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[predictionIndices, :4], tbox[targetIndices]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouVector[0]).nonzero():
                            detectedTarget = targetIndices[i[j]]  # detected target
                            if detectedTarget not in detected:
                                detected.append(detectedTarget)
                                correct[predictionIndices[j]] = ious[j] > iouVector  # iou_thres is 1xn
                                if len(detected) == numLabels:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            testStatistics.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), targetClass))

        # Plot images
        if batchIdx < 1:
            f = 'test_batch%g_gt.jpg' % batchIdx  # filename
            plot_images(imgs, targets, paths=paths, names=classNames, fname=f)  # ground truth
            f = 'test_batch%g_pred.jpg' % batchIdx
            plot_images(imgs, output_to_target(output, width, height), paths=paths, names=classNames, fname=f)  # predictions

    # Compute statistics
    testStatistics = [np.concatenate(x, 0) for x in zip(*testStatistics)]  # to numpy
    if len(testStatistics):
        precision, recall, AP, F1, APClass = ap_per_class(*testStatistics)
        if niou > 1:
            precision, recall, AP, F1 = precision[:, 0], recall[:, 0], AP.mean(1), AP[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        meanPrecision, meanRecall, mAP, meanF1 = precision.mean(), recall.mean(), AP.mean(), F1.mean()
        numTargets = np.bincount(testStatistics[3].astype(np.int64), minlength=numClasses)  # number of targets per class
    else:
        numTargets = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, numTargets.sum(), meanPrecision, meanRecall, mAP, meanF1))

    # Print results per class
    if verbose and numClasses > 1 and len(testStatistics):
        for i, currClass in enumerate(APClass):
            print(pf % (classNames[currClass], seen, numTargets[currClass], precision[i], recall[i], AP[i], F1[i]))


    # Return results
    mAPs = np.zeros(numClasses) + mAP
    for i, currClass in enumerate(APClass):
        mAPs[currClass] = AP[i]
    return (meanPrecision, meanRecall, mAP, meanF1, *(loss.cpu() / len(dataloader)).tolist()), mAPs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    # task = 'test', 'study', 'benchmark'
    test(opt.cfg, opt.data, opt.weights, opt.batch_size, opt.img_size, opt.conf_thres, opt.iou_thres, opt.augment)
