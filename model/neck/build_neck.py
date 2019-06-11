import torch.nn as nn
import torch.nn.functional as F
import sys
from ..utils import ConvModule


class Neck(nn.Module):
    def __init__(self, in_channels = [64,256,512,1024,2048],out_channels = 256,out_map=None,start_level = 0,end_level = None):
        super(Neck,self).__init__()
        self.in_channels = in_channels
        if isinstance(out_channels,int):
            out_channels = [out_channels for i in range(len(self.in_channels))]
        self.out_channels = out_channels
        
        #select the out map
        self.out_map = out_map
        self.start_level = start_level
        self.end_level = end_level
        self.normalize = {'type':'BN'}
        if self.end_level is None:
        	self.end_level = len(self.in_channels)

        if self.start_level<0 or self.end_level>len(self.in_channels):
        	assert Exception("start_level or end_level is error")


        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.end_level):
            l_conv = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                1,
                normalize=self.normalize,
                bias=False,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels[i],
                out_channels[i],
                3,
                padding=1,
                normalize=self.normalize,
                bias=False,
                inplace=True)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self,inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            
            laterals[i - 1] += F.interpolate(
                laterals[i],size = laterals[i-1].shape[2:], mode='nearest')
        # build outputs
        # part 1: from original levels
        outs = [
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]
        if self.out_map is not None:
            outs = outs[self.out_map]
        '''
        for i in range(len(outs)):
            print(outs[i].shape)
        '''
        return tuple(outs)



