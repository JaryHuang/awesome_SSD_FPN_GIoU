# config.py
import os.path
# gets home dir cross platform
#HOME = os.path.expanduser("~")
#HOME = os.path.abspath(os.path.dirname(__file__)).split("/") this path
HOME = os.path.join(os.getcwd()) #../path
# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

#crcak (104, 117, 123)
# SSD300 CONFIGS
train_config = {
    'loss_c':"CrossEntropy", #CrossEntropy,FocalLoss
    'loss_r':"CIoU",  #SmoothL1,GIoU,DIoU,CIoU
}


voc= {
    'model':"resnet50",
    'num_classes':21,
    'mean':(123.675, 116.28, 103.53),
    'std':(58.395, 57.12, 57.375),#(58.395, 57.12, 57.375),
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'max_epoch':80,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'backbone_out':[512,1024,2048,512,256,256],
    #'neck_out':[512,1024,2048,512,256,256],
    'neck_out':[256,256,256,256,256,256],
    'steps':[8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
    'nms_method':"nms",
    'work_name':"SSD300_VOC_FPN_CrossEntropy_CIoU",
}
