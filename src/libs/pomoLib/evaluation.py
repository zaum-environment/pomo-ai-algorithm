# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 12:02:42 2022

@author: stemmler_t
"""

import logging
logger = logging.getLogger("root.EvalLogger")
logger.debug("EvalLogger has been initialized")

import os
import cv2
import shutil
import numpy as np
import skimage.io
from collections import Counter
import time
import tifffile
import pandas as pd
from lxml import etree as xml
import xml.etree.ElementTree as ElementTree
import pickle

#own libs
from libs.pomoLib import pomoUtils
from libs.pomoLib.datatypes import (sampleType, pathComp, posImg, imgElemType, 
                                    dcPomoObject)
# removed synthes lib for SYLVA
from libs.pomoLib.segmentation.segmenter import PomoSegmentation
from libs.pomoLib.classifier import PomoClassification
from dateutil.parser import parse
import pytz
#from tensorflow.keras import models


class RegionAnalyzer:
    """
    This class represents a imageRegion of a given sample. With it, it is 
    possible to synthesize the image stack of the region, segmentate the 
    synth image and classify the founded objects
    """
    
    def __init__(self, pathImg: str, fgBlkSynth: bool = False, 
                 fgBlkRoughSynth: bool = False):

        self.pathImg: str = pathImg
        self.fgBlkSynth: bool = fgBlkSynth
        self.fgBlkRoughSynth: bool = fgBlkRoughSynth
        
        (self.imgStack, 
         self.imgSynth) = self.__getStackForAnalyzer()
        
        self.imgSynthBlk = None
        self.imgSynthBlkRough = None
        self.imgSeg = None
        self.posImg: posImg() = posImg
        self.imgWidth: int = None
        self.imgHeight: int = None
        self.__getImgSize()
        self.dustPart: int = 0
        self.lstPomoObjs: list = [dcPomoObject]
        
        
    @property
    def pathImg(self):
        return self._pathImg
    
    @pathImg.setter
    def pathImg(self, val):
        if val.endswith(".tif") or val.endswith(".png"):
            self._pathImg = val
        else:
            raise ValueError("Only images with .tif or .png format are allowed")
    
    def __getImgSize(self):
        if self.imgSynth is not None:
            if len(self.imgSynth.shape) == 3:
                self.imgHeight, self.imgWidth, __ = self.imgSynth.shape
            else:
                self.imgHeight, self.imgWidth = self.imgSynth.shape
                
        elif self.imgStack is not None:
            self.imgHeight, self.imgWidth = self.imgStack[0].shape
        else:
            logger.error("Could not get size of image")
            raise Exception ("Could not get size of image")
    
    def __getStackForAnalyzer(self):
        """
        This function checks if the argument is a stack or a synth image.

        Parameters
        ----------
        img : Array of uint8
            Image to be checked

        Returns
        -------
        imgStack : Array of uint8 / None
            If stack return the stack, otherwise None
        imgSynth : Array of uint8 / None
            If synth return the synth, otherwise None

        """
        perFail = False
        while(1):
            try:
                img = skimage.io.imread(self.pathImg)
            except:
                if not perFail:
                    logger.warning(f"No permission to load file{self.pathImg}")
                    perFail = True
            else:
                break
        

        if self.pathImg.endswith(".tif"):
            imgStack = img
            imgSynth = None
            return imgStack, imgSynth
            
        imgStack = None
        imgSynth = img
        return imgStack, imgSynth
    

    def synthesize(self, samplingFactor, blockSizeHalf,
                   blackSynthImg = False, blackRoughSynthImg = False):
        # If region is synth it hasn't to be synth
        if self.imgStack is None:
            return None
        
        # Removed synthesizer for SYLVA 
        
    def segmentate(self, pomoSegmenter: PomoSegmentation):
        # Convert image from gray to color (needed for segmentation)
        if not len(self.imgSynth.shape) == 3:
            self.imgSynth = cv2.cvtColor(self.imgSynth, cv2.COLOR_GRAY2RGB)
        # Segmentate object from synth img
        results = pomoSegmenter.detect(self.imgSynth, verbose=0)
        r = results[0]
        # Create masked image
        self.imgSeg = pomoSegmenter.maskedImage(self.imgSynth, r, 
                                                drawDust = False)
        # # Generate image name
        # self.nameImgSeg = self.__getImgName(imgElemType.seg.value)
        # Count of dust pariticals
        self.dustPart = pomoSegmenter.countParticle(r)   
        # Cuts spores, pollen and fragments, to get them classified afterwards
        self.lstPomoObjs = pomoSegmenter.cutObj(self.imgSynth, r, 
                                                     drawDust = False)
        
    
    def classify(self, pomoClassifier: PomoClassification):
        # Extract img from lstPomoObjs that are pollen or spores.
        # Get the list position (index) as well.
        if self.lstPomoObjs is None:
            return None
        
        imgClassif, objIndex = zip(*[(obj.imgObj, self.lstPomoObjs.index(obj)) 
                                      for obj in self.lstPomoObjs if 
                                      (obj.segClass == "pollen" or 
                                      obj.segClass == "sporen")])
        
        
        classifObj = pomoClassifier.classifyObj(imgClassif)
        
        if not len(classifObj) == len(imgClassif):
            logger.error("Something went wrong during classification!")
            raise Exception("Something went wrong during classification!")
        
        # Extract classif results 
        for i, obj in zip(range(len(objIndex)), classifObj):

            self.lstPomoObjs[objIndex[i]].specFolder = obj["species"].nameGer
            
            self.lstPomoObjs[objIndex[i]].clfScore = obj["hitRate"]
            self.lstPomoObjs[objIndex[i]].clfSpeciesInt = obj["index"]
            self.lstPomoObjs[objIndex[i]].clfSpecies = obj["species"]
            
            self.lstPomoObjs[objIndex[i]].clfScoreSec = obj["hitRateSecond"]
            self.lstPomoObjs[objIndex[i]].clfSpeciesIntSec = obj["indexSecond"]
            self.lstPomoObjs[objIndex[i]].clfSpeciesSec = obj["speciesSecond"]

                 
    def createLittleStack(self, pathLittleStack):
        offset = 10
        
        for obj in self.lstPomoObjs:
            # Don't save fragment and NoPollen
            if obj.specFolder == "Fragment" or obj.specFolder == "NoPollen":
                continue
            
            #Get offset (border of 10 px around the mask)
            x = pomoUtils.getOffset(obj.xPos, -offset, 0)  
            y = pomoUtils.getOffset(obj.yPos, -offset, 0)
            width = pomoUtils.getOffset(obj.width, offset*2, self.imgWidth)
            height = pomoUtils.getOffset(obj.height, offset*2, self.imgHeight)
            
            # Change endian of name
            name = obj.imgObj.split(".png")[0]
            name += ".tif"
            
            with tifffile.TiffWriter(os.path.join(pathLittleStack,
                                                  obj.specFolder,
                                                  name)) as stack:    
                for img in self.imgStack:
                    stack.save(img[y:y+height, x:x+width], 
                               photometric='minisblack', contiguous=True)
                    
            #Save name in instance        
            obj.imgObjStack = name
            
class Evaluator:
    """ 
    This class represents a sample.
    """
    
    def __init__(self, sampleInfo: dict, segmenter, classifier, 
                 pathSampleFolder: str, pathEvalOut: str, pathOutAnalysis: str, 
                 saveLittleStacks: bool, evalSynthOnly: bool, saveStacks: int,
                 volStrom: int, carrierTypePlastic: bool, deviceType: str, 
                 deviceName: str, serialNumber: str, version:str):
        
        self.sampleType: sampleType = sampleInfo.get("sampleType")
        self.segmenter = segmenter
        self.classifier = classifier
        self.pathSampleFolder = pathSampleFolder
        self.pathEvalOut = pathEvalOut
        self.pathOutAnalysis = pathOutAnalysis
        self.saveLittleStacks = saveLittleStacks
        self.evalSynthOnly = evalSynthOnly
        self.saveStacks = saveStacks
        self.volStrom = volStrom
        self.carrierType = "Glass"
        if carrierTypePlastic:
            self.carrierType = "Plastic"
        
        self.deviceName = deviceName
        self.serialNumber = serialNumber
        
        self.deviceType = deviceType
        self.versionPomoAI = version
        
        self.versionClassif = classifier._modelVersion
        self.versionSegment = segmenter._modelVersion
        self.active = True
        self.endOfSample = False
        self.evalPast = False
        self.dustPartTotal = 0
        self.lstDYT = []
        
        self.activePomoAI: bool = False
        self.pathStatusASC: str = None
        self.nameSample: str = None
        self.activeImgRegion: str = None
        self.numExistingRegions = 0
        self.lstCreatedSpecFolder = []
        self.lstImgRegions: list[RegionAnalyzer] = []
        self.regIter = None  #Cont. name of region of already eval sample
        self.beginnDerProbenahme: str = None 
        self.endeDerProbenahme: str = None 
        self.stationNumber: str = None
        self.unzippedPathSample: str = None

        # Get name of element of sample
        if not self.sampleType == sampleType.file:
            # Unzip sample if zipped
            if self.sampleType == sampleType.zipped:
                logger.info("Unzip sample")
                
                pathZipFile = os.path.join(self.pathSampleFolder, 
                                           sampleInfo.get("pathSample"))
                
                pathSample = pomoUtils.unzipProbe(pathZipFile)
                self.unzippedPathSample = pathSample

            elif self.sampleType == sampleType.folder:
                pathSample = sampleInfo.get("pathSample")
            
            self.pathImgRegionFolder = os.path.join(self.pathSampleFolder, 
                                                    pathSample, 
                                                    "images")
            
                
            # Check if analysis file exists. Means sample has been evaluated
            for elem in os.listdir((os.path.join(self.pathSampleFolder, 
                                                 pathSample, 
                                                 "analysis"))):
                
                if "xml.xml" in elem:
                    print("Processing xml: ", elem)
                    
                    xml_root = ElementTree.parse(os.path.join(self.pathSampleFolder, 
                                                      pathSample, "analysis",
                                                      elem))                   
                        
                    if (xml_root.find("./Device") is not None and xml_root.find("./Device").text):
                        self.deviceName = xml_root.find("./Device").text
                    
                    if (xml_root.find("./Analysenvolumenstrom") is not None and xml_root.find("./Analysenvolumenstrom").text):
                        self.volStrom =  int(float(xml_root.find("./Analysenvolumenstrom").text))
                    
                    if (xml_root.find("./Seriennummer") is not None and xml_root.find("./Seriennummer").text):
                        self.serialNumber = xml_root.find("./Seriennummer").text
                        
                    if (xml_root.find("./Beginn_der_Probenahme") is not None and xml_root.find("./Beginn_der_Probenahme").text):
                        self.beginnDerProbenahme = xml_root.find("./Beginn_der_Probenahme").text
                        
                    if (xml_root.find("./Ende_der_Probenahme") is not None and xml_root.find("./Ende_der_Probenahme").text):
                        self.endeDerProbenahme = xml_root.find("./Ende_der_Probenahme").text
                           
                    if (xml_root.find("./WMO-Stationsnummer") is not None and xml_root.find("./WMO-Stationsnummer").text):
                        self.stationNumber = xml_root.find("./WMO-Stationsnummer").text
                        
                        
                if "asc.txt" in elem:
                    self.evalPast = True
                    self.pathStatusASC = os.path.join(self.pathSampleFolder, 
                                                      pathSample, "analysis",
                                                      elem)
            
                    # Get name and number of regions
                    lstNameRegions = [elem for elem in 
                                      os.listdir(self.pathImgRegionFolder)
                                      if elem.endswith("SYN._FP.png") or 
                                      elem.endswith("SYN.png")]
                    
                    self.regIter = iter(lstNameRegions)
                    self.numExistingRegions = len(lstNameRegions)
                    sampleString = lstNameRegions[0]
            
            # Sample is curently evaluated by FIT
            if not self.evalPast:
                #sampleString .....
                pass
            
            
            self.nameSample, __ = os.path.splitext(sampleInfo.get("pathSample"))
            
        else:
            sampleString = sampleInfo.get("pathSample")
            
            self.pathImgRegionFolder = self.pathSampleFolder
        
        # Extract infomation from file name
        sampleInfoSplit = pomoUtils.getPathInfo(sampleString)
        
        self.sampleDateTime = sampleInfoSplit[pathComp.dateTime.value]
        
        self.dateYear = self.sampleDateTime[0:4]
        self.dateMonth = self.sampleDateTime[4:6]
        self.dateDay = self.sampleDateTime[6:8]
        self.dateHour = self.sampleDateTime[8:10]
        self.dateMin = self.sampleDateTime[10:12]
        self.dateSec = self.sampleDateTime[12:14]
        
        self.device = sampleInfoSplit[pathComp.device.value]
        self.barcode = sampleInfoSplit[pathComp.barcode.value]
        
        if not self.nameSample:
           self.nameSample = (self.sampleDateTime + "_" + self.barcode)
          
        self.__pathOutImg: str = None
        self.__pathOutLittleStack: str = None
        self.__pathOutClassif: str = None
        self.__pathOutAnalysis: str = None
        self._pathOutTemp: str = None
        
        # Create the output dir
        self.__createSampleOutputFolder() 
        
        if self.pathStatusASC is not None:
            tail, head = os.path.split(self.pathStatusASC)
            
            shutil.copyfile(self.pathStatusASC, os.path.join(self.__pathOutAnalysis,
                                                             head))
        
        logger.info(f"Starting evaluation of sample: {self.nameSample}")
        
    # def getPollenTreshold(self, pathPollenTreshold):
    #     logger.info("Load Treshold")
        
    def __createSampleOutputFolder(self):
        logger.debug("Create Output directory")
        #Create main output directory
        pathOut = os.path.join(self.pathEvalOut, self.nameSample)
        #Check wether old output folder of probe exist. (unexpeced interrupt)
        try:
            if os.path.isdir(pathOut):
                logger.info('Deleting existing output dir')
                shutil.rmtree(pathOut)
        except PermissionError:
            logger.error("Could not delete sample folder. Please close all files!")
            raise PermissionError("Could not delete sample folder. Please close all files!")
            
            
        os.makedirs(pathOut)
        
        #Create structure of output directory
        self.__pathOutImg = os.path.join(pathOut, "images")
        if not os.path.isdir(self.__pathOutImg):
            os.mkdir(self.__pathOutImg)
            
        if self.saveLittleStacks:
            self.__pathOutLittleStack = os.path.join(pathOut, 'LittleStacks')
            if not os.path.isdir(self.__pathOutLittleStack):
                os.mkdir(self.__pathOutLittleStack)
            
        self.__pathOutClassif = os.path.join(pathOut, "pollen_DL")
        if not os.path.isdir(self.__pathOutClassif):
            os.mkdir(self.__pathOutClassif)
        
        self.__pathOutAnalysis = os.path.join(pathOut, "analysis")
        if not os.path.isdir(self.__pathOutAnalysis):
            os.mkdir(self.__pathOutAnalysis)
            
        self.__pathOutCsv = os.path.join(pathOut, "csv")
        if not os.path.isdir(self.__pathOutCsv):
            os.mkdir(self.__pathOutCsv)
            
        self._pathOutTemp = os.path.join(pathOut, 'temp')
        if not os.path.isdir(self._pathOutTemp):
            os.mkdir(self._pathOutTemp)
    
    def nextImageRegion(self):
        self.activeImgRegion = None
        # Standalone PomoAI looking at "D:/Pollenmonitor/TestInDir"
        if self.sampleType == sampleType.file:
            # Loocking for new image region
            for elem in os.listdir(self.pathSampleFolder):
                
                if not elem.endswith(".tif") and not elem.endswith("asc.txt"):
                    continue
                    
                # Check if element is from current sample
                pathInfo = pomoUtils.getPathInfo(elem)
                if not pathInfo:
                    logging.info(f"{elem} got wrong string format")
                    continue
                
                if not(pathInfo[pathComp.barcode.value] == self.barcode and
                    pathInfo[pathComp.dateTime.value] == self.sampleDateTime):
                    continue
                
                # if asc.txt file -> end of sample
                if pathInfo[pathComp.elemType.value].endswith("asc.txt"):
                    sourcePath = os.path.join(self.pathSampleFolder, elem)
                    destPath = os.path.join(self.__pathOutAnalysis, elem)
                    
                    #Check if other no .tif is left in InDir
                    elemLeft = False
                    for elemTemp in os.listdir(self.pathSampleFolder):
                        if elemTemp.endswith(".tif"):
                            # Check if element is from current sample
                            pathInfo = pomoUtils.getPathInfo(elemTemp)
                            if not pathInfo:
                                logging.info(f"{elemLeft} got wrong string format")
                                continue
                            
                            if not(pathInfo[pathComp.barcode.value] == self.barcode and
                                pathInfo[pathComp.dateTime.value] == self.sampleDateTime):
                                continue
                            
                            elemLeft = True
                        
                    if elemLeft == True:
                        continue
                    
                    # Wait until file is accessible due to copying of file
                    perFail = False
                    while(1):
                        try:
                            shutil.copyfile(sourcePath, destPath)
                            os.remove(sourcePath)
                        except:
                            if not perFail:
                                logger.warning(f"No permission to load file{self.pathStatusASC}")
                                perFail = True
                        else:
                            break
                    
                    self.pathStatusASC = os.path.join(destPath)
                    
                    self.endOfSample = True
                    return None
                
                self.activeImgRegion = elem
                self.activePomoAI = True
                break
        
        # Parallel evaluation looking at e.g. "D:/Pollenmonitor/ResultDir" 
        else:
            # Reanalyse an existing sample
            if self.regIter:
                # Get next region
                self.activeImgRegion = next(self.regIter, None)
                
                # If no region left -> end of sample
                if not self.activeImgRegion:
                    self.endOfSample = True
                    return None
            
            else:
                sampleElems = [elem for 
                               elem in 
                               os.listdir(os.path.join(self.pathImgRegionFolder))
                               if elem.endswith("SYN.png") or 
                                  elem.endswith("SYN._FP.png")]
                
                
                for elem in sampleElems:
                    if not self.lstImgRegions:
                        self.activeImgRegion = elem
                        break
                    else:    
                        for reg in self.lstImgRegions:
                            if not elem in reg.pathImg:
                                self.activeImgRegion = elem
                                break
                        if self.activeImgRegion:
                            break
                        
                if not self.activeImgRegion:
                    return None
            
            
            # try to load img stack if evalSynthOnly is disabled
            if not self.evalSynthOnly:
                
                stackPath = pomoUtils.modPath(self.activeImgRegion, 
                                              pathComp.elemType.value, 
                                              "tiff.tif")
                
                if stackPath:
                    if os.path.isfile(os.path.join(self.pathSampleFolder,
                                                   self.nameSample,
                                                   "images",
                                                   stackPath)):
                        self.activeImgRegion = stackPath
                    else:
                        comp = pomoUtils.getPathInfo(self.activeImgRegion)
                        logger.debug("Could not find stack of pos: "
                                       f"{comp[pathComp.imgRegPos.value]}. " 
                                       "Using synth img instead")
                else:
                    logger.error("Wrong string format")
                    raise ValueError("Wrong string format")
                
                
        if self.activeImgRegion:
            pathNewImageRegion = os.path.join(self.pathImgRegionFolder, 
                                              self.activeImgRegion)           
              
            # Create instance of imageAnalyser and save add it to lstImgRegion
            ImgRegion = RegionAnalyzer(pathNewImageRegion)
            return ImgRegion
        else:
            self.active = False
            return None
                    
    def analyzeRegion(self, imgReg: RegionAnalyzer):
        if not isinstance(imgReg, RegionAnalyzer):
            raise TypeError("Given object is no 'RegionAnalyzer'")
        
        head, tail = os.path.split(imgReg.pathImg)
        if self.numExistingRegions != 0:
            logger.info(f"Analyse region({len(self.lstImgRegions) + 1}/"
                        f"{self.numExistingRegions}): {tail}")
        else:
            logger.info(f"Analyse region({len(self.lstImgRegions) + 1}):"
                        f" {tail}")
        
        # Extract position of img region
        try:
            info = pomoUtils.getPathInfo(imgReg.pathImg)
            position = info[pathComp.imgRegPos.value].split("_")
            imgReg.posImg =  posImg(position[2], position[3], position[1])
        except:
            imgReg.posImg = posImg("01", "01", "01")
        
        # Synthesize
        start = time.time()
        imgReg.synthesize(samplingFactor = 4, blockSizeHalf = 6)
        end = time.time()
        logger.info(f"Synthezise ({(end - start):.0f} s)")
        
        
        # Segmentate
        start = time.time()
        imgReg.segmentate(self.segmenter)
        end = time.time()
        logger.info(f"Segmentate ({(end - start):.0f} s)")
        
        
        if (imgReg.lstPomoObjs is not None) and any(((i.segClass == "pollen") or 
                                                     (i.segClass == "sporen")) 
                                                    for i in imgReg.lstPomoObjs):
          
            # Classify objects
            start = time.time()
            imgReg.classify(self.classifier)
            end = time.time()
            logger.info(f"Classify ({(end - start):.0f} s)")
            
            species = []
            for obj in imgReg.lstPomoObjs:
                # if object is undefined (Score lower than Threshold)
                if obj.clfScore < obj.clfSpecies.default.score:
                    obj.specFolder = "Undefined"
                    obj.sortedOut = "Undefined"
                
            
                # Create output folder 
                if obj.specFolder not in self.lstCreatedSpecFolder:
                    logger.debug(f"Crete output folder for: {obj.specFolder}")
                    try:
                        os.mkdir(os.path.join(self.__pathOutClassif, 
                                              obj.specFolder))
                    except FileExistsError:
                        pass
                    #Create output folder for little Stacks
                    if self.saveLittleStacks:
                        if (not obj.specFolder == "Fragment" and 
                            not obj.specFolder == "NoPollen"):    
                            try:
                                os.mkdir(os.path.join(self.__pathOutLittleStack, 
                                                      obj.specFolder)) 
                            except FileExistsError:
                                pass
                            
                    self.lstCreatedSpecFolder.append(obj.specFolder)
                    
                # Create img name
                objImgName = ("obj_img" + "-" + 
                              str(imgReg.posImg.zPos) + "_" + 
                              str(imgReg.posImg.xPos) + "_" + 
                              str(imgReg.posImg.yPos) + "-" +
                              self.sampleDateTime + "-" +
                              "pmon" + "-" +
                              self.device + "-" +
                              self.barcode + "-" +
                              obj.clfSpecies.nameGer + "-" +
                              str(obj.clfScore) + "-" +
                              str(obj.xPos) + "_" +
                              str(obj.yPos) + "_" +
                              str(obj.width) + "_" +
                              str(obj.height))
                
                if obj.specFolder == "Undefined":
                    objImgName += "-Unf"
                    
                objImgName += ".png"
                    
                species.append(obj.specFolder)
                
                # Save image
                logger.debug(f"Save image {objImgName}")
                cv2.imwrite(os.path.join(self.__pathOutClassif,
                                          obj.specFolder, objImgName), 
                            obj.imgObj)
                
                obj.imgObj = objImgName
                
            species = Counter(species)   
            outString = "Found:"
            for key, value in species.items():
                if key != "Fragment" and key != "Undefined" and key != "NoPollen":
                    
                    outString += f" {value}x{key};"
                    
            # Output the number of found objects
            if outString != "Found:":
                logger.info(outString[:-1])
            else:
                logger.info("No objects found")
            
            # Save little stacks
            if self.saveLittleStacks and imgReg.imgStack is None:
                logger.debug("Could not save littleStacks. Input file is not"
                             " a stack.")
            elif self.saveLittleStacks:      
                logger.info("Save little stacks")
                # Create little stacks
                imgReg.createLittleStack(self.__pathOutLittleStack)
        else:
            logger.info("No objects found")
        
            
        # imgReg.getAvgGreyOfStack()
        
        
        # Generate synth image name
        nameImgSynth = self.__getImgName(imgReg.pathImg,
                                         imgElemType.synth.value)
        
        if len(imgReg.imgSynth.shape) == 3:
            imgReg.imgSynth = cv2.cvtColor(imgReg.imgSynth, cv2.COLOR_RGB2GRAY)
        
        # Save synthesized image
        logger.debug("Save synth image")
        cv2.imwrite(os.path.join(self.__pathOutImg, nameImgSynth), 
                    imgReg.imgSynth)
        
        imgReg.imgSynth = nameImgSynth
        
        # Generate seg image name
        nameImgSeg = self.__getImgName(imgReg.pathImg,
                                       imgElemType.seg.value)
        
        # Save segmented image
        logger.debug("Save segmented image")
        cv2.imwrite(os.path.join(self.__pathOutImg, nameImgSeg), imgReg.imgSeg)
        
        imgReg.imgSeg = nameImgSeg
            
        
        # Generate stack image name
        nameImgStack = self.__getImgName(imgReg.pathImg, imgElemType.tif.value)
        
        
        # Save img stack 
        if ((self.saveStacks == 1 and imgReg.imgStack is not None) or
            (self.saveStacks == 2 and imgReg.lstPomoObjs is not None)):

            logger.debug("Save stack image")
            skimage.io.imsave(os.path.join(self.__pathOutImg, nameImgStack), 
                              imgReg.imgStack)
           
        imgReg.imgStack = nameImgStack
        
        # Add found count of partical to total count
        self.dustPartTotal += imgReg.dustPart
        
        # Save instace of img region
        self.lstImgRegions.append(imgReg)
        
        # save instance of img region to temp folder in case of unexpeced 
        # programm error
        if self.activePomoAI:
            with open(os.path.join(self._pathOutTemp,
                                   (self.sampleDateTime + "_" + 
                                    self.barcode + "_" + 
                                    imgReg.posImg.zPos + "_" + 
                                    imgReg.posImg.xPos + "_" +
                                    imgReg.posImg.yPos + ".pkl")), "wb") as f:
                
                pickle.dump(imgReg, f)
        
        # Last action: delete img stack
        if self.activePomoAI:
            logger.info("Remove img stack from sample input folder")
            os.remove(os.path.join(self.pathSampleFolder, 
                                   self.activeImgRegion))
            

    def __computeTreshold(self):
        logger.debug("Compute Treshold")
        
        arten = [species.specFolder for reg in self.lstImgRegions if 
                 reg.lstPomoObjs is not None for species in reg.lstPomoObjs ]
        
        arten = Counter(arten)
        
        # Dynamic treshold
        for spec, count in arten.items():
            if (spec == "Fragment" or spec == "Undefined" or 
                spec == "NoPollen" or spec == "Gammel"):
                continue
            
            objs = [species for reg in self.lstImgRegions if reg.lstPomoObjs is not None 
                    for species in reg.lstPomoObjs if species.specFolder == spec]
            
            numTresh = objs[0].clfSpecies.default.num
            scoreMax = objs[0].clfSpecies.default.scoreMax
            
            # If num of objects for the class is create than the species num, 
            # than the class is not effected by the DYT
            if count > numTresh:
                continue
            
            # check if treshold of all objects is greater than maxScore -> DYT has no effect
            if all(i.clfScore >= scoreMax for i in objs):
                continue
            
            # 4PL Regression (Sigmoid)
            zeroConc = objs[0].clfSpecies.default.score
            infConc = scoreMax
            midConc = numTresh / 2
            slopeFactor = -6
            
            # Calculate minimum threshold for obj class (depends on num of class)            
            minScore = infConc + ((zeroConc - infConc) / 
                                  (1 + pow((count / midConc), slopeFactor)))
            
            logger.debug(f"Minimal score for {spec} is {minScore}")
            
            
            # check if treshold of all objects is greater than minScore
            if all(i.clfScore >= minScore for i in objs):
                continue
            
            # Create undefined folder if not exist
            if "Undefined" not in self.lstCreatedSpecFolder:
                logger.debug("Crete output folder for: Undefined")
                try:
                    os.mkdir(os.path.join(self.__pathOutClassif, "Undefined"))
                except FileExistsError:
                    pass
                
                #Create output folder for little Stacks
                if self.saveLittleStacks:
                    try:
                        os.mkdir(os.path.join(self.__pathOutLittleStack, 
                                              "Undefined"))
                    except FileExistsError:
                        pass
                    
                self.lstCreatedSpecFolder.append("Undefined")
            
            #list of DYT params for csv file
            self.lstDYT.append({"species" : spec,
                                "minScore" : minScore,
                                "count" : count,
                                "zeroConc" : zeroConc,
                                "infConc" : infConc,
                                "maxCount" : numTresh,
                                "slopeFactor" : slopeFactor})
            
            for obj in objs:
                
                # Get img name and change endian
                sourceImgName = obj.imgObj
                destImgName = sourceImgName.split(".png")[0]
                destImgName += "-DYT.png"
                
                obj.sortedOut = "DYT"
                
                sourceStackName = None
                if obj.imgObjStack is not None:
                    sourceStackName = obj.imgObjStack
                    destStackName = sourceStackName.split(".tif")[0]
                    destStackName += "-DYT.tif"
                
                
                specFolderTemp = obj.specFolder
                # Copy img from original species folder to undefined folder
                sourceImg = os.path.join(self.__pathOutClassif, 
                                         specFolderTemp,
                                         sourceImgName)
                
                obj.specFolder = "Undefined"
                destImg = os.path.join(self.__pathOutClassif, obj.specFolder,
                                       destImgName)
                
                try:
                    shutil.copy(sourceImg, destImg)
                except:
                    logger.error(f"Error during copy of {sourceImg} "
                                 f"to {destImg}")
                    
                    
                # Copy stack from original species folder to undefined folder
                if self.saveLittleStacks and obj.imgObjStack is not None:
                    sourceStack = os.path.join(self.__pathOutLittleStack, 
                                               specFolderTemp, 
                                               sourceStackName)
    
                    destStack = os.path.join(self.__pathOutLittleStack, 
                                             obj.specFolder, destStackName)
        
                    try:
                        shutil.copy(sourceStack, destStack)
                    except:
                        logger.error(f"Error during copy of {sourceStack} "
                                     f"to {destStack}")
                    
            
            # Remove species folder
            shutil.rmtree(os.path.split(sourceImg)[0])
            
            if self.saveLittleStacks:   
                shutil.rmtree(os.path.join(self.__pathOutLittleStack, 
                                           specFolderTemp))
    
    
    
    def createAnalysisFiles(self):
        logger.info("Create analyse files")
        
        # Enable for sylva
        #self.__createJsonOuputFile()
        
        # Desable for sylva
        self.__createXmlAnalysisFile()
        self.__createCsvFile()
        
        
        
    def sampleEnd(self):
        logger.info("End of sample reached")
        
        
        self.__computeTreshold()
        
        self.createAnalysisFiles()
        
        #start = time.time()
        #self.makeMap()
        #end = time.time()
        #logger.info(f"MakeMap ({(end - start):.0f} s)")
        
        #if self.sampleType == sampleType.zipped:
            #logger.debug("Delete unzippt sample folder")
        folder_path = os.path.join(self.pathSampleFolder, self.nameSample)
        if os.path.isdir(folder_path):
            shutil.rmtree(os.path.join(self.pathSampleFolder, self.nameSample))

        
        # remove temp folder (should be last action)
        logger.debug("Remove temp folder from sample")
        shutil.rmtree(self._pathOutTemp)
        
        folder_to_zip = os.path.join(self.pathEvalOut, self.nameSample)
        zip_file_path = shutil.make_archive(folder_to_zip, 'zip', folder_to_zip)
        
        zip_file_name = os.path.basename(zip_file_path)
        
        destination_path = os.path.join(self.pathOutAnalysis, zip_file_name)
        shutil.move(zip_file_path, destination_path)
        
        shutil.rmtree(folder_to_zip)
        shutil.rmtree(self.unzippedPathSample, ignore_errors=True)

        print(f"Folder '{folder_to_zip}' zipped and moved to '{destination_path}'")
        
        self.active = False
        
        
    def makeMap(self):
        """
        Creates a map of the evaluated sample carrier. In total 24x24 img regiopns.
        For those regions without img, a black field will be included

        Returns
        -------
        None.

        """
        if self.lstImgRegions is None:
            logger.info("Could not create map")
            return None
        
        logger.info("Creating map of sample carrier")
        
        imgH = self.lstImgRegions[0].imgHeight
        imgW = self.lstImgRegions[0].imgWidth
        
        newImgH = int(imgH / 2)
        newImgW = int(imgW / 2)
        
        sampleMap = np.zeros((newImgH*24,newImgW*24), dtype=np.uint8)
        
        for imgReg in self.lstImgRegions:
            xPos = int(imgReg.posImg.xPos)
            yPos = int(imgReg.posImg.yPos)
            
            imgTemp = cv2.imread(os.path.join(self.pathSampleFolder, self.__pathOutImg, 
                                 imgReg.imgSynth), cv2.IMREAD_GRAYSCALE)
            
            imgTemp = cv2.resize(imgTemp, (newImgW, newImgH), 
                                 interpolation = cv2.INTER_AREA)
            
            sampleMap[yPos*newImgH : yPos*newImgH + newImgH, 
                      xPos*newImgW : xPos*newImgW + newImgW] = imgTemp
        
            
        cv2.imwrite(os.path.join(self.__pathOutImg, self.nameSample + "_Map.png"),
                    sampleMap)

    
    def __getImgName(self, pathImg, imgType:list):
        head, tail = os.path.split(pathImg)
        # If img is part of an sample
        if pomoUtils.getPathInfo(tail):
            imgName = pomoUtils.modPath(tail, 
                                        pathComp.elemType.value, 
                                        imgType)
        # if img is not part of an sample (RegionAnalyser standalone)
        else:    
            file, ext = os.path.splitext(tail)
            imgName = file + imgType
            
        return (imgName)
    
    
    def __createCsvFile(self):
     
        f = open(os.path.join(self.__pathOutCsv, (self.sampleDateTime + "_" + 
                                                  self.barcode + '_01.csv')), "a")
        
        f.write("x;y;z;Width;Height;YearRecorded;MonthRecorded;DayRecorded;"
                "HourRecorded;MinuteRecorded;ProbeDirName;ImageName;"
                "Synthetic_Image;GraphCut_Image;ImageStackProgrammCall;"
                "SegMask;SegPrediction;SegScore;SegManuel;"
                "PredictedPollenSpecies;PredictedPollenSpeciesLatin;SortedOut;"
                "PredictionReliability;PollenSpecies;SubClass;PredictedSubClass;"
                "NameSW;PollenMonitorVersion;SegVersion;ClassifVersion;"
                "Device;SerialNumber;DeviceType;CarrierType;Comment\n")
        
        for region in self.lstImgRegions:
            if not region.lstPomoObjs:
                continue
            for obj in region.lstPomoObjs:
                if obj.specFolder == "NoPollen":
                    continue
                
                xOff = pomoUtils.getOffset(obj.xPos, -10, 0)
                yOff = pomoUtils.getOffset(obj.yPos, -10, 0)
                wOff = pomoUtils.getOffset(obj.width, 10*2, region.imgWidth)
                hOff = pomoUtils.getOffset(obj.height, 10*2, region.imgHeight)
                
                imgSynth = ("%..\images\\" + region.imgSynth + 
                            f"?{xOff},{yOff},{wOff},{hOff}")
                
                graphCut = (f"%..\images\{region.imgSeg}?{xOff},{yOff},"
                            f"{wOff},{hOff}")
            
                imgStackProgCall = ("~|Bildstapel|D:\Pollenmonitor\\"
                                    "VisualisationProgramm\\"
                                    "PollenVisualisation.exe|..\images\\"
                                    f"{region.imgSynth} {xOff} {yOff} "
                                    f"{wOff} {hOff}|")
                
                flattenMask = obj.segMask.flatten().tolist()
                flattenMask = str(flattenMask).replace("[","").replace("]","").replace(" ","").replace(",","") 
                
                sortedOut = "--"
                if obj.sortedOut:
                    sortedOut = obj.sortedOut
                
                subClass = "--"
                if obj.clfSpecies.subClass:
                    subClass = obj.clfSpecies.subClass
                
                f.write(f"{obj.xPos};{obj.yPos};{obj.zPos};{obj.width};"
                        f"{obj.height};{self.dateYear};{self.dateMonth};"
                        f"{self.dateDay};{self.dateHour};{self.dateMin};"
                        f"{self.nameSample};{region.imgStack};{imgSynth};"
                        f"{graphCut};{imgStackProgCall};{flattenMask};"
                        f"{obj.segClass};{obj.segScore:.4f};--;"
                        f"{obj.clfSpecies.nameGer};{obj.clfSpecies.nameLat};"
                        f"{sortedOut};{obj.clfScore:.4f};--;{subClass}"
                        f";--;PomoAI;{self.versionPomoAI};{self.versionSegment};"
                        f"{self.versionClassif};{self.deviceName};{self.serialNumber};"
                        f"{self.deviceType};{self.carrierType};--\n")
                
        f.close()

    
    
    
    def __createXmlAnalysisFile(self):
        
        logger.debug("Create xml file")
        
        dictStatus = self.__readTxtStatus()
        
        xmlFileName = ("polle-ad_01-" + self.sampleDateTime + "-" +
                       "pmon-" + self.device + "-" + self.barcode + "-xml.xml")
                
        xmlFile = self.__createXmlFile(xmlFileName)
        myTree = xml.parse(xmlFile)
        myRoot = myTree.getroot()
        
        sampleTime = int(dictStatus['Probenahmezeit'])
        anzahlBilder = int(dictStatus['Anzahl_gescannte_Bilder'])
        
        if anzahlBilder != len(self.lstImgRegions):
            logger.info("Differences in number of evaluated image"
                        " and number in status file")
        
        
        analysedSampleVolume = ((sampleTime * self.volStrom * 
                                 len(self.lstImgRegions)) / 
                                (60 * 490))
        
        IntakeVol = (sampleTime / 60) * self.volStrom
        
        Concentrationlist = xml.SubElement(myRoot, "Konzentrationsliste")
        
        arten = [species.specFolder for reg in self.lstImgRegions if 
                 reg.lstPomoObjs is not None for species in reg.lstPomoObjs ]
        
        arten.sort()
        arten = Counter(arten)
        
        for spec, count in arten.items():
            if (spec == "Fragment" or spec == "Undefined" or 
                spec == "NoPollen" or spec == "Gammel"):
                continue
            
            objs = [species for reg in self.lstImgRegions if reg.lstPomoObjs is not None 
                    for species in reg.lstPomoObjs if species.specFolder == spec]
            
            try:
                pollenConc = str(round((count * 1000) / 
                                       (0.6 * 0.8 * analysedSampleVolume),4))
            except:
                pollenConc = str(0)
                logger.error('Pollenkonzentration konnte nicht ermittelt werden!')
            
            xml.SubElement(Concentrationlist, "Konzentrationsinformation",
                           Deutscher_Name_Pollenart = objs[0].clfSpecies.nameGer,
                           Lateinischer_Name_Pollenart = objs[0].clfSpecies.nameLat,
                           Pollenanzahl = str(count), 
                           Pollenkonzentration = pollenConc)
            
            Pollenlist = xml.SubElement(myRoot,"Pollenliste", 
                                        Lateinischer_Name_Pollenart = objs[0].clfSpecies.nameLat)
            
            for obj in objs:
                region = [reg.imgStack for reg in self.lstImgRegions 
                          if (reg.lstPomoObjs is not None and 
                              obj in reg.lstPomoObjs)]
                
        
                xml.SubElement(Pollenlist,"Polle", 
                               Probenbild = region[0],
                               Qualitaetsmass = f'{obj.clfScore:.4f}',
                               x_Ortskoordinate=f'{obj.xPos}',
                               y_Ortskoordinate=f'{obj.yPos}', 
                               z_Ortskoordinate=f"{obj.zPos}")
        
            
        dytList = xml.SubElement(myRoot, "Dynamischer_Treshold")
        for dyt in self.lstDYT:
            xml.SubElement(dytList,"DYT", 
                           Pollenart = f'{dyt["species"]}', 
                           Gefunden = f'{dyt["count"]}',
                           MinScore = f'{dyt["minScore"]:.4f}',
                           GW_Unten = f'{dyt["zeroConc"]:.4f}',
                           GW_Oben = f'{dyt["infConc"]:.4f}',
                           Anzahl_Max = f'{dyt["maxCount"]:.4f}',
                           Steigung = f'{dyt["slopeFactor"]:.4f}')    
        
        
        for key, value in dictStatus.items():
            xml.SubElement(myRoot, key).text = value
        
        
        # Analyse data
        xml.SubElement(myRoot, "Analysenvolumenstrom").text = str(round(self.volStrom, 4))
        xml.SubElement(myRoot, "Einsaugvolumen").text = str(round(IntakeVol, 4))
        xml.SubElement(myRoot, "Analysiertes_Probevolumen").text = str(round(analysedSampleVolume, 4))
        
        
        # Number of dust particles
        xml.SubElement(myRoot, "Anzahl_Partikel").text = str(self.dustPartTotal)
        
        # Device data
        xml.SubElement(myRoot, "Device").text = self.deviceName 
        xml.SubElement(myRoot, "Seriennummer").text = self.serialNumber 
        
        # Version of PomoAI, segmenter and classifier
        xml.SubElement(myRoot, "Name_SW").text = "PomoAI"
        xml.SubElement(myRoot, "Version_PomoAI").text = self.versionPomoAI
        xml.SubElement(myRoot, "Version_Segmentierer").text = self.versionSegment
        xml.SubElement(myRoot, "Version_Klassifikator").text = self.versionClassif
    
        # Write to xml file
        myTree.write(xmlFile, pretty_print=True)
        
        if len(self.pathOutAnalysis) != 0:
            if os.path.isdir(self.pathOutAnalysis):
                logger.debug(f"Save analysis file to {self.pathOutAnalysis}")
                head, tail = os.path.split(xmlFile)
                #shutil.copyfile(xmlFile, os.path.join(self.pathOutAnalysis, tail))
            else:
                logger.error(f"Analysis output folder does not exist({self.pathOutAnalysis})")
        
        
    def __createXmlFile(self, filename):
        pathFile = os.path.join(self.__pathOutAnalysis, filename)
        if not os.path.exists(pathFile):
            with open((pathFile), "ab") as f:
                root = xml.Element("Analyse_Datensatz")
                tree = xml.ElementTree(root)
                tree.write(pathFile, encoding = 'UTF-8', xml_declaration = True )
        return pathFile
        
    
    def __readTxtStatus(self):
        perFail = False
        while(1):
            try:
                df = pd.read_csv(self.pathStatusASC, sep=';', 
                                 usecols=range(0, 2), header=None).to_dict()
            except:
                if not perFail:
                    logger.warning(f"No permission to load file{self.pathStatusASC}")
                    perFail = True
            else:
                break
        
        
        return {df[0][i]:df[1][i] for i in range(len(df[0]))}
    
    
    
    def __convert_datetime_timezone(self, dt, from_time_zone, to_time_zone):
        from_time_zone = pytz.timezone(from_time_zone)
        to_time_zone = pytz.timezone(to_time_zone)

        dt = from_time_zone.localize(dt)
        dt = dt.astimezone(to_time_zone)
        dt = dt.strftime("%Y-%m-%d %H:%M:%S")
        return dt    

    def __timestamp_to_datetime(self, timestamp: str):    
        """Returns a given timestamp string from BAA500 XML file as a datetime (for example datetime.datetime(2023, 9, 10, 3, 0, 3))."""
        ts = parse(timestamp)
    
        # Sometime, the instrument writes as start time "10:00:03" or "10:01:10" we set minutes and seconds back to 0 
        if ts.time().minute <= 10:
            ts = ts.replace(minute=0)
        
        ts = ts.replace(second=0)

        ts = self.__convert_datetime_timezone(ts, "Europe/Berlin", "UTC")
        ts = parse(ts)
        return ts.timestamp()
        
    def __createJsonOuputFile(self):
        
        logger.debug("Create JSON output file")
        
        dictStatus = self.__readTxtStatus()
        
        jsonFileName = ("polle-ad_01-" + self.sampleDateTime + "-" +
                       "pmon-" + self.device + "-" + self.barcode + "-json.json")
        
        
        sampleTime = int(dictStatus['Probenahmezeit'])
        anzahlBilder = int(dictStatus['Anzahl_gescannte_Bilder'])
        
        if anzahlBilder != len(self.lstImgRegions):
            logger.info("Differences in number of evaluated image"
                        " and number in status file")
        
        
        analysedSampleVolume = ((sampleTime * self.volStrom * 
                                 len(self.lstImgRegions)) / 
                                (60 * 490))
        
        IntakeVol = (sampleTime / 60) * self.volStrom
        
        arten = [species.specFolder for reg in self.lstImgRegions if 
                 reg.lstPomoObjs is not None for species in reg.lstPomoObjs ]
        
        arten.sort()
        arten = Counter(arten)
        
        pollen_list = []
        
        for spec, count in arten.items():
            if (spec == "Fragment" or spec == "Undefined" or 
                spec == "NoPollen" or spec == "Gammel"):
                continue
            
            objs = [species for reg in self.lstImgRegions if reg.lstPomoObjs is not None 
                    for species in reg.lstPomoObjs if species.specFolder == spec]
            
            try:
                pollenConc = str(round((count * 1000) / 
                                       (0.6 * 0.8 * analysedSampleVolume),4))
            except:
                pollenConc = str(0)
                logger.error('Pollenkonzentration konnte nicht ermittelt werden!')
            
            pollen_list.append ({"name": objs[0].clfSpecies.nameLat, "concentration": pollenConc, "uncertainty": 0})
           
        
            
        
        device = [{'id':self.stationNumber,'serial_number': self.serialNumber,'software_version':self.versionPomoAI}]
        
                                
        utc_time_start = self.__timestamp_to_datetime(self.beginnDerProbenahme)
        utc_time_end = self.__timestamp_to_datetime(self.endeDerProbenahme)
         

        data = [{'start': utc_time_start,
                'end': utc_time_end,
                'device': pd.Series(device)[0],
                'pollen':pd.Series(pollen_list)
                }]
        
        pd_to_json = pd.DataFrame(data)
        
        if len(self.pathOutAnalysis) != 0:
            if os.path.isdir(self.pathOutAnalysis):
                logger.debug(f"Save JSON file to {self.pathOutAnalysis}")
                pd_to_json.to_json(os.path.join(self.pathOutAnalysis, jsonFileName), orient="records", lines=True)
            else:
                logger.error(f"JSON output folder does not exist({self.pathOutAnalysis})")
        
            