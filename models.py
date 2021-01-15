from utils.parse_config import *
from utils.utils import *
import torch.nn.functional as F

class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class MixConv2d(nn.Module):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()

        groups = len(k)

        # 'equal_params': equal parameter count per group
        b = [out_ch] + [0] * groups
        a = np.eye(groups + 1, groups, k=-1)
        a -= np.roll(a, 1, axis=1)
        a *= np.array(k) ** 2
        a[0] = 1
        ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch, out_channels=ch[g], kernel_size=k[g], stride=stride, padding=k[g] // 2, dilation=dilation, bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)

def create_modules(module_defs, imgSize, cfg):
    # Constructs module list of layer blocks from module configuration in module_defs

    imgSize = [imgSize] * 2 if isinstance(imgSize, int) else imgSize  # expand if necessary
    trainingHyperparms = module_defs.pop(0)  # cfg training hyperparams (unused)
    outputFilters = [3]  # input channels
    moduleList = nn.ModuleList()
    routingLayers = []  # list of layers which rout to deeper layers
    yoloIndex = -1

    for idx, currModule in enumerate(module_defs):
        modules = nn.Sequential()

        if currModule['type'] == 'convolutional':
            isBatchNormalize = currModule['batch_normalize']
            filters = currModule['filters']
            kernelSize = currModule['size']  # kernel size
            stride = currModule['stride'] if 'stride' in currModule else (currModule['stride_y'], currModule['stride_x'])
            
            if isinstance(kernelSize, int):  # single-size conv
                modules.add_module('Conv2d', nn.Conv2d(in_channels=outputFilters[-1], out_channels=filters, kernel_size=kernelSize, stride=stride, padding=kernelSize // 2 if currModule['pad'] else 0, groups=currModule['groups'] if 'groups' in currModule else 1, bias=not isBatchNormalize))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=outputFilters[-1], out_ch=filters, k=kernelSize, stride=stride, bias=not isBatchNormalize))

            if isBatchNormalize:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routingLayers.append(idx)  # detection output (goes into yolo layer)

            if currModule['activation'] == 'leaky':  
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

        elif currModule['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=currModule['stride'])

        elif currModule['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = currModule['layers']
            filters = sum([outputFilters[l + 1 if l > 0 else l] for l in layers])
            routingLayers.extend([idx + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif currModule['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = currModule['from']
            filters = outputFilters[-1]
            routingLayers.extend([idx + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in currModule)

        elif currModule['type'] == 'yolo':
            yoloIndex += 1
            stride = [32, 16, 8]  # P5, P4, P3 strides
            layers = currModule['from'] if 'from' in currModule else []
            modules = YOLOLayer(anchors=currModule['anchors'][currModule['mask']],  # anchor list
                                nc=currModule['classes'],  # number of classes
                                img_size=imgSize,  # (416, 416)
                                yolo_index=yoloIndex,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride[yoloIndex])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            j = layers[yoloIndex] if 'from' in currModule else -1

            bias_ = moduleList[j][0].bias  # shape(255,)
            bias = bias_[:modules.numOutputs * modules.numAnchors].view(modules.numAnchors, -1)  # shape(3,85)
            bias[:, 4] += -4.5  # obj
            bias[:, 5:] += math.log(0.6 / (modules.numClasses - 0.99))  # cls (sigmoid(p) = 1/nc)
            moduleList[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)


        # Register module list and number of output filters
        moduleList.append(modules)
        outputFilters.append(filters)

    binaryRoutingLayers = [False] * (idx + 1)
    for idx in routingLayers:
        binaryRoutingLayers[idx] = True
    return moduleList, binaryRoutingLayers


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.numOutputLayers = len(layers)  # number of output layers (3)
        self.numAnchors = len(anchors)  # number of anchors (3)
        self.numClasses = nc  # number of classes (80)
        self.numOutputs = nc + 5  # number of outputs (85)
        self.numX, self.numY, self.numGridpoints = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchorVector = self.anchors / self.stride
        self.anchorWH = self.anchorVector.view(1, self.numAnchors, 1, 1, 2)


    def create_grids(self, ng=(13, 13), device='cpu'):
        self.numX, self.numY = ng  # x and y grid size
        self.numGridpoints = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.numY, device=device), torch.arange(self.numX, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.numY, self.numX, 2)).float()

        if self.anchorVector.device != device:
            self.anchorVector = self.anchorVector.to(device)
            self.anchorWH = self.anchorWH.to(device)

    def forward(self, p, out):

        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.numX, self.numY) != (nx, ny):
            self.create_grids((nx, ny), p.device)

        p = p.view(bs, self.numAnchors, self.numOutputs, self.numY, self.numX).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            inferenceOutput = p.clone()  # inference output
            inferenceOutput[..., :2] = torch.sigmoid(inferenceOutput[..., :2]) + self.grid  # xy
            inferenceOutput[..., 2:4] = torch.exp(inferenceOutput[..., 2:4]) * self.anchorWH  # wh yolo method
            inferenceOutput[..., :4] *= self.stride
            torch.sigmoid_(inferenceOutput[..., 4:])
            return inferenceOutput.view(bs, -1, self.numOutputs), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = get_yolo_layers(self)
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x, augment=False, verbose=False):

        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []


        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:  # sum, concat
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])

        if self.training:  # train
            return yolo_out
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
 
def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]


