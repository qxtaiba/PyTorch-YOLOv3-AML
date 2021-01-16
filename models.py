from utils.parse_config import *
from utils.utils import *
import torch.nn.functional as F

class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layerIndices = layers  
        self.isMultipleLayers = len(layers) > 1  

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layerIndices], 1) if self.isMultipleLayers else outputs[self.layerIndices[0]]


class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
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
            
            modules.add_module('Conv2d', nn.Conv2d(in_channels = outputFilters[-1], out_channels = filters, kernel_size = kernelSize, stride = stride, padding = kernelSize // 2 if currModule['pad'] else 0, groups = currModule['groups'] if 'groups' in currModule else 1, bias = not isBatchNormalize))

            if isBatchNormalize:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum = 0.03, eps = 1E-4))
            else:
                routingLayers.append(idx)  # detection output (goes into yolo layer)

            if currModule['activation'] == 'leaky':  
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace = True))

        elif currModule['type'] == 'upsample':
            modules = nn.Upsample(scale_factor = currModule['stride'])

        elif currModule['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = currModule['layers']
            filters = sum([outputFilters[l + 1 if l > 0 else l] for l in layers])
            routingLayers.extend([idx + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers = layers)

        elif currModule['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = currModule['from']
            filters = outputFilters[-1]
            routingLayers.extend([idx + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers = layers, weight ='weights_type' in currModule)

        elif currModule['type'] == 'yolo':
            yoloIndex += 1
            stride = [32, 16, 8]  # P5, P4, P3 strides
            layers = currModule['from'] if 'from' in currModule else []
            modules = YOLOLayer(anchors = currModule['anchors'][currModule['mask']],  # anchor list
                                nc = currModule['classes'],  # number of classes
                                img_size = imgSize,  # (416, 416)
                                yolo_index = yoloIndex,  # 0, 1, 2...
                                layers = layers,  # output layers
                                stride = stride[yoloIndex])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            j = layers[yoloIndex] if 'from' in currModule else -1

            bias_ = moduleList[j][0].bias  # shape(255,)
            bias = bias_[:modules.numOutputs * modules.numAnchors].view(modules.numAnchors, -1)  # shape(3,85)
            bias[:, 4] += -4.5  # obj
            bias[:, 5:] += math.log(0.6 / (modules.numClasses - 0.99))  # cls (sigmoid(p) = 1/nc)
            moduleList[j][0].bias = torch.nn.Parameter(bias_, requires_grad = bias_.requires_grad)


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
        self.layerIndex = yolo_index  
        self.layerIndices = layers  
        self.layerStride = stride  
        self.numOutputLayers = len(layers)  
        self.numAnchors = len(anchors) 
        self.numClasses = nc  
        self.numOutputs = nc + 5  
        self.numX, self.numY, self.numGridpoints = 0, 0, 0  
        self.anchorVector = self.anchors / self.layerStride
        self.anchorWH = self.anchorVector.view(1, self.numAnchors, 1, 1, 2)


    def create_grids(self, ng =(13, 13), device ='cpu'):
        self.numX, self.numY = ng  
        self.numGridpoints = torch.tensor(ng, dtype = torch.float)

        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.numY, device = device), torch.arange(self.numX, device = device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.numY, self.numX, 2)).float()

        if self.anchorVector.device != device:
            self.anchorVector = self.anchorVector.to(device)
            self.anchorWH = self.anchorWH.to(device)

    def forward(self, prediction, out):

        bs, _, ny, nx = prediction.shape  # bs, 255, 13, 13
        if (self.numX, self.numY) != (nx, ny):
            self.create_grids((nx, ny), prediction.device)

        prediction = prediction.view(bs, self.numAnchors, self.numOutputs, self.numY, self.numX).permute(0, 1, 3, 4, 2).contiguous()  

        if self.training:
            return prediction

        else:
            inferenceOutput = prediction.clone() 
            inferenceOutput[..., :2] = torch.sigmoid(inferenceOutput[..., :2]) + self.grid  # xy
            inferenceOutput[..., 2:4] = torch.exp(inferenceOutput[..., 2:4]) * self.anchorWH  # wh yolo method
            inferenceOutput[..., :4] *= self.layerStride
            torch.sigmoid_(inferenceOutput[..., 4:])
            return inferenceOutput.view(bs, -1, self.numOutputs), prediction  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size =(416, 416), verbose = False):
        super(Darknet, self).__init__()

        self.moduleDefinitions = parse_model_cfg(cfg)
        self.moduleList, self.routs = create_modules(self.moduleDefinitions, img_size, cfg)
        self.yoloLayers = get_yolo_layers(self)
        self.version = np.array([0, 2, 5], dtype = np.int32)  
        self.numImageSeen = np.array([0], dtype = np.int64)  

    def forward(self, x, augment = False, verbose = False):

        if not augment:
            return self.forward_once(x)
        else:  
            imageSize = x.shape[-2:]  
            scales = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x, torch_utils.scale_img(x.flip(3), scales[0], same_shape = False),  torch_utils.scale_img(x, scales[1], same_shape = False))):
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= scales[0]  # scale
            y[1][..., 0] = imageSize[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= scales[1]  # scale

            y = torch.cat(y, 1)

            return y, None

    def forward_once(self, inferenceOutput, augment = False):
        imageSize = inferenceOutput.shape[-2:]  
        yoloLayerOutput, output = [], []

        for i, module in enumerate(self.moduleList):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']: 
                inferenceOutput = module(inferenceOutput, output)  
            elif name == 'YOLOLayer':
                yoloLayerOutput.append(module(inferenceOutput, output))
            else: 
                inferenceOutput = module(inferenceOutput)

            output.append(inferenceOutput if self.routs[i] else [])

        if self.training:
            return yoloLayerOutput
        else: 
            inferenceOutput, trainingOutput = zip(*yoloLayerOutput)  
            inferenceOutput = torch.cat(inferenceOutput, 1)  
            if augment:  
                # de-augment results
                inferenceOutput = torch.split(inferenceOutput, nb, dim=0)
                # scale
                inferenceOutput[1][..., :4] /= s[0]
                # flip lr
                inferenceOutput[1][..., 0] = imageSize[1] - inferenceOutput[1][..., 0] 
                # scale
                inferenceOutput[2][..., :4] /= s[1]  

                inferenceOutput = torch.cat(inferenceOutput, 1)

            return inferenceOutput, trainingOutput

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fuseList = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fuseList.append(a)
        self.moduleList = fuseList
 
def get_yolo_layers(model):
    return [i for i, m in enumerate(model.moduleList) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]

def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

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
    for idx, (moduleDef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if moduleDef['type'] == 'convolutional':
            conv = module[0]
            if moduleDef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


