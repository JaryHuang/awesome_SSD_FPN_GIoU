"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os
import sys
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from .VOC import VOCDetection,VOCAnnotationTransform




CRACK_CLASSES = (  # always index 0
    'neg',)

HOME = os.path.join(os.getcwd())
CRACK_ROOT = os.path.join(HOME, "data/CrackData/")

class CRACKDetection(VOCDetection):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root = CRACK_ROOT,
                image_sets= 'trainval.txt',transform=None, 
                bbox_transform = VOCAnnotationTransform(class_to_ind = CRACK_CLASSES),
                dataset_name = 'CRACK'):
        self.root = root
        self.transform = transform
        self.bbox = bbox_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        rootpath = os.path.join(self.root, 'crack/')
        for line in open(os.path.join(rootpath, 'ImageSets', 'Main',image_sets)):
            self.ids.append((rootpath, line.strip()))
        #self.ids = self.ids[0:20]


    def mix_up(self,fir_index):
        fir_id = self.ids[fir_index]
        sec_index = np.random.randint(0,len(self.ids))
        sec_id = self.ids[sec_index]
        first_img = cv2.imread(self._imgpath % fir_id)
        second_img = cv2.imread(self._imgpath % sec_id)
        if eq(first_img.shape,second_img.shape) is False:
            raise Exception("The image shape is not same, please!,the first img is {},shape = {},\
                the second is {},shape = {}".format(fir_index,str(first_img.shape),sec_index,str(second_img.shape)))
            
        else:
            height,width,channels = first_img.shape
        
        lam = np.random.beta(1.5, 1.5)
        res = cv2.addWeighted(first_img, lam, second_img, 1-lam,0)
        
        first_target = ET.parse(self._annopath % fir_id).getroot()
        second_target = ET.parse(self._annopath % sec_id).getroot()

        target = []
        if lam <= 0.9 and lam >= 0.1:
            target = self.target_transform(first_target, width, height)
            target+= self.target_transform(second_target, width, height)
        elif lam>0.9:
            target = self.target_transform(first_target, width, height)
        else:
            target = self.target_transform(second_target, width, height)
        return res,target,height,width
    

    

    

    