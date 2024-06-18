# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:26:02 2023

@author: stemmler_t
"""

import logging
logger = logging.getLogger("root.SegLogger")
logger.debug("SegLogger has been initialized")


import os
import sys

import numpy as np
import cv2

#import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Root directory of the project
ROOT_DIR = os.path.join(os.getcwd(), "segmentation")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from libs.pomoLib.segmentation.config import Config
from libs.pomoLib.segmentation import visualize as visualize
#from libs.pomoLib.segmentation.model import MaskRCNN

#sys.path.append("../")
from libs.pomoLib.datatypes import dcPomoObject, dcSpecies, dcTreshold
from libs.pomoLib.segmentation import model as segModel


class PollenConfig(Config):
    
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "pollen"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # BG + fiber, particle, pollen, sporen, pollen_fragment

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 188
    VALIDATION_STEPS = 81

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7
    
    BACKBONE = "resnet50"
    
    MEAN_PIXEL = np.array([140, 140, 140])
    
    #IMAGE_RESIZE_MODE = "crop"
    #IMAGE_MIN_DIM = 512
    #IMAGE_MAX_DIM = 512
    #IMAGE_MIN_SCALE = 2.0
    
    IMAGE_MIN_DIM = 1280
    IMAGE_MAX_DIM = 960
    #IMAGE_MIN_SCALE = 2.0
    
    LEARNING_RATE = 0.001
    
    RPN_ANCHOR_SCALES = (16,32, 64, 128, 256)
    

# class InferenceConfig(PollenConfig):    
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
    

class PomoSegmentation:
    
    def __init__(self, pathModel, segConfig: PollenConfig, classNames: list):
        self.pathModel = pathModel
        self.classNames = classNames
        self.segConfig = segConfig
        
        if len(classNames) != self.segConfig.NUM_CLASSES:
            logger.exception("Number of classes from config not equal to seg classes")
            raise ValueError("Number of classes from config not equal to seg classes")
        
        logger.debug("Setup segmenter model to inference mode")
        self.model = segModel.MaskRCNN(mode = "inference", 
                                       config = self.segConfig, 
                                       model_dir = "./")
        
        if os.path.isfile(pathModel):
            # Get version
            head, tail = os.path.split(self.pathModel)
            version = tail.split("_")[0]
            if version.startswith("v"):
                self._modelVersion = version
            else:
                logger.warning("Could not get version of classifier")
                self._modelVersion = "vX.X"
                
            logger.info(f"Load segmentation model: {tail}")
            self.model.load_weights(pathModel, by_name=True)
            
        else:
            logger.error("File/path of segmenter model is not existing")
            raise ValueError("File/path of segmenter model is not existing")
            
        
    
    #def __init__(self, segModel: MaskRCNN, classNames: list):
       #self.segModel = segModel
       
       
       
       
    def detect(self, images, verbose=0):
        return self.model.detect([images], verbose = 0)
    
    def maskedImage(self, synthImage, r, drawDust = False):
        return visualize.display_instances(synthImage, r['rois'], r['masks'],
                                           r['class_ids'], self.classNames, 
                                           drawDust, r['scores'],)
    
    def countParticle(self, r):
            intCountParticle = 0
            
            for mask in range(len(r['masks'][0][0])):
                if self.classNames[r['class_ids'][mask]] == 'particle':
                    intCountParticle += 1
            
            return intCountParticle
                                                              
    def cutObj(self, synthImage, r, drawDust = False):
        try:
            masks = [[r['masks'][:,:,mask], r['rois'][mask], 
                      self.classNames[r['class_ids'][mask]], r['scores'][mask]]  
                     for mask in range(len(r['masks'][0][0])) if 
                     ((self.classNames[r['class_ids'][mask]] == 'pollen') or
                      (self.classNames[r['class_ids'][mask]] == 'sporen') or 
                      (self.classNames[r['class_ids'][mask]] == 'fragment')or 
                      (self.classNames[r['class_ids'][mask]] == 'particle' if 
                       drawDust else None))]
            
        except:
            return None
        
        if not masks:
            return None
        
        cutObjs: dcPomoObject = []
        for index, mask in enumerate(masks):
            x0 = mask[1][1]
            x1 = mask[1][3]
            y0 = mask[1][0]
            y1 = mask[1][2]
            
            if x1 - x0 > 350:
                diff = round(((x1-x0) - 350) / 2)
                x0 = x0 + diff
                x1 = x0 + 350
            
            if y1 - y0 > 350:
                diff = round(((y1-y0) - 350) / 2)
                y0 = y0 + diff
                y1 = y0 + 350
            
            polle = mask[0][y0:y1,x0:x1]
            
            polle = polle.astype('uint8')
               
            imgTemp = synthImage[y0:y1,x0:x1] 
            polleTemp = cv2.bitwise_and(imgTemp,imgTemp,mask=polle)
            polleTemp = cv2.cvtColor(polleTemp,cv2.COLOR_RGB2GRAY)
            
            l_img = np.zeros((350,350))
            s_img = polleTemp
            x_offset= int((350 - polleTemp.shape[1])/2)
            y_offset= int((350 - polleTemp.shape[0])/2)
            try:
                l_img[y_offset:y_offset+s_img.shape[0],
                      x_offset:x_offset+s_img.shape[1],] = s_img
            except:
                return None
            
            obj = dcPomoObject(xPos = x0,
                               yPos = y0,
                               width = x1 - x0,
                               height = y1 - y0,
                               imgObj = l_img.astype(dtype="uint8"),
                               segMask = polle,
                               segClass = mask[2],
                               segScore = mask[3])
            
            if mask[2] == 'fragment':
                obj.specFolder = "Fragment"
                obj.clfSpecies = dcSpecies("Fragment", "Fragment", "Fragment","",
                                           dcTreshold(1,0,0))
                obj.clfScore = 1.00
                
            # check if object on image region is double masked (added 15.11.2023 by T. Stemmler)
            doublePos = [elem for elem in cutObjs if (
                         elem.xPos == obj.xPos and
                         elem.yPos == obj.yPos and
                         elem.width == obj.width and
                         elem.height == obj.height)]
            
            if doublePos:
                doublePos = doublePos[0]
                if doublePos.segScore > obj.segScore:
                    pass
                else:
                    cutObjs[cutObjs.index(doublePos)] = obj  
                continue
            
            cutObjs.append(obj)
                               
        return cutObjs



if __name__ == '__main__':
    synthetic_image_file_path = "input\\test.png" 
    save_path = "output"
    
    #main_segmentation_pipeline(synthetic_image_file_path, save_path)
    



    










































