import argparse

from models import *
from utils.datasets import *
from utils.utils import *


def detect():
    imgsz = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights = opt.output, opt.source, opt.weights

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    with open(opt.names, 'r') as f:
        lines = f.read().split('\n')
    names = list(filter(None, lines))  # filter removes empty strings (such as last line)
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.float()) 
    for path, img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to 32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for idx, detections in enumerate(pred):  # detections for image i
            p, prtStr, im0 = path, '', im0s

            saveDir = str(Path(out) / Path(p).name)
            prtStr += '%gx%g ' % img.shape[2:]  # print string

            if detections is not None and len(detections):
                # Rescale boxes from imgsz to im0 size
                detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], im0.shape).round()

                # Print results
                for c in detections[:, -1].unique():
                    n = (detections[:, -1] == c).sum()  # detections per class
                    prtStr += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(detections):
                    # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Stream results
            cv2.imshow(p, im0)

            # Save results (image with detections)
            cv2.imwrite(saveDir, im0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
