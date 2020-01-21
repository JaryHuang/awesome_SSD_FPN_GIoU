'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import voc, coco
import os
import torchvision
'''
from model.backbone import Backbone
from model.neck import Neck,SSDNeck
from model.head import SSDHead
import torch.nn as nn
import torch
from utils import PriorBox,Detect
import torch.nn.functional as F
from torchsummary import summary

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base ResNet followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        Backbone: some network like ResNet layers for input, size of 300
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        cfg: some config information to initialition the priorbox
    """

    def __init__(self, phase, size, Backbone, Neck, Head, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()  #some priors[-1,4]
        self.size = size
        # SSD network
        self.backbone = Backbone
        self.neck = Neck
        self.head = Head
        self.num_classes = cfg['num_classes']
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(self.num_classes , 0, 200, 0.01, 0.45, variance = cfg['variance'],nms_method=cfg['nms_method'])

    def forward(self, x, phase):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
            phase: now is not used
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        

        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)

        conf,loc = self.head(x)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                #self.priors.type(type(x.data))                  # default boxes
                self.priors
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_ssd(phase, size=300, cfg=None):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300 and size!=600 and size!=800:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    print(phase)
    #the layer [6,7,8,9,10,11] is the feature map of 38 19 10 5 3 1
    base = Backbone(cfg['model'],[6,7,8,9,10,11])
    #base.cuda()
    #summary(base, (3,300, 300))
    #neck=None
    neck = Neck(in_channels = cfg['backbone_out'], out_channels = cfg['neck_out'])
    head = SSDHead(num_classes = cfg['num_classes'],in_channels = cfg['neck_out'],aspect_ratios = cfg['aspect_ratios'])
    
    return SSD(phase, size, base, neck, head, cfg)
