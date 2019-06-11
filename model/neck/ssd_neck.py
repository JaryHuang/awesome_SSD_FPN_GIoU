import torch.nn as nn
import torch.nn.functional as F
from ..utils import ConvModule


class SSDNeck(nn.Module):
    def __init__(self, in_channels = [1024,2048],out_channels = 256,out_map=None,start_level = 0,end_level = None):
        super(SSDNeck,self).__init__()
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
        	self.end_level = len(self.out_channels)


        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.end_level):
            if i == 0 :
                fpn_conv = ConvModule(
                    in_channels[-1],
                    out_channels[0],
                    3,
                    stride = 2,
                    padding=1,
                    normalize=self.normalize,
                    bias=True,
                    inplace=True)
            else:
                fpn_conv = ConvModule(
                    out_channels[i-1],
                    out_channels[i],
                    3,
                    stride = 1,
                    padding=0,
                    normalize=self.normalize,
                    bias=True,
                    inplace=True)

            self.fpn_convs.append(fpn_conv)

        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self,inputs):
        assert len(inputs) == len(self.in_channels)

        outs = []
        # build outputs
        # part 1: from original levels
        
        x = inputs[-1]
        outs += inputs
        for i in range(self.start_level, self.end_level):
            x = self.fpn_convs[i](x)
            outs.append(x)   
        if self.out_map is not None:
            outs = outs[self.out_map]
        '''
        for i in range(len(outs)):
            print(outs[i].shape)
        '''
        return tuple(outs)



