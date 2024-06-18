# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 08:31:47 2023

@author: stemmler_t
"""


"""
--------------------------------------
IMPORT LIBS
--------------------------------------

"""

import logging
logger = logging.getLogger("root.ClassifLogger")
logger.debug("ClassifLogger has been initialized")

import cv2
import numpy as np
from tensorflow.keras import models
import os


"""
--------------------------------------
Classes
--------------------------------------

"""

class PomoClassification:
    
    def __init__(self, pathModel):
        self.pathModel = pathModel
        
        self.speciesNames: list = []
        
        if os.path.isfile(pathModel):
            # Get version
            head, tail = os.path.split(self.pathModel)
            version = tail.split("_")[0]
            if version.startswith("v"):
                self._modelVersion = version
            else:
                logger.warning("Could not get version of classifier")
                self._modelVersion = "vX.X"
            logger.info(f"Load classifier model: {tail}")
            # Load model
            self._model = models.load_model(pathModel)
            
        else:
            logger.error("File/path of classifier model is not existing")
            raise ValueError("File/path of classifier model is not existing")
            
        # Check if classNames numb same as class num in model
        
    def setSpeciesNames(self, Spec:list):
        self.speciesNames = Spec
             

    def classifyObj(self, cuttedObj, print_klassifikation = False):
        vgg19_image_size = 350
        
        numImg = len(cuttedObj)
        imgPred= np.zeros((numImg, vgg19_image_size,vgg19_image_size,3), 
                          np.float32)
        
        for i, obj in zip(range(numImg), cuttedObj):
            shape = obj.shape

            if(len(shape) == 2):
                obj = cv2.cvtColor(obj,cv2.COLOR_GRAY2RGB)
                 
            pollePrep = np.float32(obj) / 255.0
            
            imgPred[i] = pollePrep
            

        result = self._model.predict(imgPred)
        
        classifiedObjs = []
        for r in result:
            
            indexFirst = np.argsort(r)[-1]
            classifScore = round(r[indexFirst],4)
            
            indexSecond = np.argsort(r)[-2]
            classifScloreSecond = round(r[indexSecond],4)
            
            classifiedObjs.append({'species' : self.speciesNames[indexFirst], 
                                   'index' : indexFirst, 
                                   'hitRate' : classifScore, 
                                   'speciesSecond' : self.speciesNames[indexSecond], 
                                   'indexSecond' : indexSecond, 
                                   'hitRateSecond' : classifScloreSecond})
            
            
        return classifiedObjs
          


if __name__ == '__main__':
    
    classif = PomoClassification(r"")
    classif.loadClassNames(r"")
    
    path = r""
    
    for img in os.listdir(path):
        if os.path.isfile(os.path.join(path,img)):
            img = cv2.imread(os.path.join(path,img))
            res = classif.classifyObj([img])
            res = res[0]
            
            cv2.imwrite(os.path.join(path,"out", res["species"]["ger"] + "_" + str(res["hitRate"]) + ".png"), img)
