# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 08:30:26 2022

@author: stemmler_t
"""


import logging
logger = logging.getLogger("root.AppLogger")
logger.debug("AppLogger has been initialized")


import os 
from configparser import ConfigParser
import zipfile
import pickle
import copy

# own libs
from libs.pomoLib import evaluation
from libs.pomoLib.datatypes import sampleType, dcSpecies, dcTreshold, pathComp
from libs.pomoLib import pomoUtils 

from libs.pomoLib.segmentation import segmenter
from libs.pomoLib import classifier
 

class PomoAI:
    def __init__(self, config:ConfigParser,version):
        self.config = config
        self.version = version
        
        logger.info(f"Initialize PomoAI {self.version}")
        
        # Create evaluatedSamples if ParallelEvaluation is True
        if not os.path.exists('evaluatedSamples.txt'):
            logger.debug("Create evaluatedSamples.txt")
            with open('evaluatedSamples.txt', 'w'): pass
        
        self.seg = self.initSeg()
        self.classif = self.initClassif()
        
        self.lstOpenSamples: list[evaluation.Evaluator] = []
        
        
        
    def checkForNewSample(self):
        """
        Check if there is a new sample to evaluate and check if it is a folder,
        a file or a zipfile

        Returns
        -------
        dict
            If new sample foud, the function returns a dict containing a variable
            that represent the type (folder, file or zipfile) and the name of 
            the sample

        """
        try:
            # Check if any element in folder
           if len(os.listdir(self.config.get('MAIN', 'PathSamplesIn'))):
                # Load evaluated samples
                with open('evaluatedSamples.txt', 'r', encoding="utf-8") as fd:
                    listEvaluatedSamples = [line.replace('\n', '') for line in fd]
                    
                for elem in os.listdir(self.config.get('MAIN', 'PathSamplesIn')):
                    # check wether elem contains character that looks like a dot in the middle. If so sample is corrupted
                    if ("" in elem) or ("_______" in elem):
                        continue
                
                    # Check if sample already has been evaluated
                    root, ext = os.path.splitext(elem)
                    if root in listEvaluatedSamples:
                        continue
                    
                    # Check if sample is folder, raw stack or zipped folder
                    if os.path.isdir(os.path.join(self.config.get("MAIN", "PathSamplesIn"), elem)):
                        # Check if folder is a sample folder
                        tempPath = os.path.join(self.config.get("MAIN", "PathSamplesIn"), 
                                                elem, "analysis")
                        
                        if not os.path.isdir(os.path.join(tempPath)):
                            continue

                        
                        ascFile = [file for file in os.listdir(tempPath) if
                                   file.endswith("asc.txt")]
                        
                        if not ascFile:
                            logger.debug("Sample without analysis file. Could not evaluate!")
                            continue
                            
                            
                        tempPath = os.path.join(self.config.get("MAIN", "PathSamplesIn"), 
                                                elem, "images")
                        
                        if not os.path.isdir(os.path.join(tempPath)):
                            logger.debug("Sample without images folder. Could not evaluate!")
                            continue
                            
                        return {"sampleType": sampleType.folder, "pathSample": elem}
                    
                        
                    elif elem.endswith(".tif") or elem.endswith("asc.txt"):
                        
                        stringSplit = pomoUtils.getPathInfo(elem)
                        
                        # Check if elem is part of an unfinished sample
                        for sample in self.lstOpenSamples:
                            if (sample.barcode == stringSplit[pathComp.barcode.value] and
                                sample.sampleDateTime == stringSplit[pathComp.dateTime.value] and
                                sample.device == stringSplit[pathComp.device.value]):
                                
                                    logger.info(f"Continue evaluation of sample {sample.nameSample}")
                                    sample.active = True
                                    return None
                        
                        # Check if interrupted sample (unexpected closing of PomoAI)
                        logger.debug("Check for interrupted sample")
                        
                        sampleName = (f"{stringSplit[pathComp.dateTime.value]}"
                                      f"_{stringSplit[pathComp.barcode.value]}")
                        
                        pathEvalOut = self.config.get("MAIN", "PathEvalOut")
                        
                        if ((sampleName in os.listdir(pathEvalOut)) and 
                            ("temp" in os.listdir(os.path.join(pathEvalOut,
                                                               sampleName)))):
                            
                            reloadedSample = self.reloadSample(pathEvalOut, 
                                                               sampleName)
                            
                            # Add loaded sample to active list
                            self.lstOpenSamples.append(reloadedSample)
                            return None
                            
                            
                        return {"sampleType": sampleType.file, "pathSample": elem}
                        
                    elif zipfile.is_zipfile(os.path.join(self.config.get("MAIN", "PathSamplesIn"), elem)):
                        with zipfile.ZipFile(os.path.join(self.config.get("MAIN", "PathSamplesIn"), elem), 'r') as f:
                            for name in f.namelist():
                                if name.endswith('/'):
                                    folder_name = name.split('/')[0]
                                    if folder_name + "/analysis/" in f.namelist():
                                        return {"sampleType": sampleType.zipped, "pathSample": elem}

                            
        except FileNotFoundError as e:
            logger.error(e)
            
    def loadPollenTreshold(self, pathPollenTreshold):
        if not os.path.isfile(pathPollenTreshold):
            logger.error(f"Could not find {pathPollenTreshold}")
            raise ValueError(f"Could not find {pathPollenTreshold}")
        
        threshold = ConfigParser()
        # Load original key, not lower case
        threshold.optionxform = str
        threshold.read(pathPollenTreshold)
        
        pollenSpecies = []
        for key, value in threshold.items("TRESHOLD"):
            parts = key.split(";")
            
            if len(parts) != 2:
                logger.error(f"Wrong format in treshold.ini for {key}")
                raise ValueError(f"Wrong format in treshold.ini for {key}")
                
            keySplit = parts[0].split(",")
            if len(keySplit) != 3:
                logger.error(f"Wrong format in treshold.ini for {key}")
                raise ValueError(f"Wrong format in treshold.ini for {key}")
                
            valueSplit = value.split(";")
            if len(valueSplit) != 3:
                logger.error(f"Wrong format in treshold.ini for {value}")
                raise ValueError(f"Wrong format in treshold.ini for {value}")
                
            pollenSpecies.append(dcSpecies(keySplit[0],                         # German
                                           keySplit[1],                         # Latin
                                           keySplit[2],                         # Englisch
                                           parts[1],                            # SubClass
                                           dcTreshold(int(valueSplit[0]),       # Number effected by the DYT
                                                      float(valueSplit[1]),     # Min Score for DYT or normal Treshold
                                                      float(valueSplit[2]))))   # Max Score for DYT
        return pollenSpecies
            
    def createNewSampleEvaluator(self, sampleInfo):
        # Read config for saving of image stacks
        if not 0 <= self.config.getint("MAIN", "SaveStacks") <= 2:
            logger.error("SaveStack in config.ini must be between 0-2")
            raise ValueError("SaveStack in config.ini must be between 0-2")
            
        # Create evaluator instance for sample
        evaluator = evaluation.Evaluator(sampleInfo, 
                                         self.seg,
                                         self.classif,
                                         self.config.get("MAIN", "PathSamplesIn"),
                                         self.config.get("MAIN", "PathEvalOut"),
                                         self.config.get("MAIN", "PathOutAnalysis"),
                                         self.config.getboolean("MAIN", "SaveLittleStacks"),
                                         self.config.getboolean("MAIN", "EvalSynthOnly"),
                                         self.config.getint("MAIN", "SaveStacks"),
                                         self.config.getint("MAIN", "VolumenStromPumpe"),
                                         self.config.getboolean("MAIN", "CarrierTypePlastic"),
                                         self.config.get("MAIN", "DeviceType"),
                                         self.config.get("MAIN", "Name"),
                                         self.config.get("MAIN", "SerialNumber"),
                                         self.version)
        
        self.saveSamplePickle(evaluator)
            
        # Add sample to active list     
        self.lstOpenSamples.append(evaluator)
        
    
    def saveSamplePickle(self, evaluator):
        logger.debug("Save sample in temp folder")
        # Save instance as pickle in case of unexpeced errors
        # Can't pickle segmenter and classifier. Save path of model instead                                   
        evalTemp = copy.copy(evaluator)
        evalTemp.segmenter = evalTemp.segmenter.pathModel
        evalTemp.classifier = evalTemp.classifier.pathModel
        
        with open(os.path.join(evalTemp._pathOutTemp, 
                               evalTemp.sampleDateTime + "_" +
                               evalTemp.barcode + "_sample.pkl"), "wb") as f:
            pickle.dump(evalTemp, f)
            
        del evalTemp
    
    def reloadSample(self, pathEvalOut, sampleName):
        logger.info(f"Reload sample {sampleName}")
        # Load pickled instance of sample
        pathPickles = os.path.join(pathEvalOut, sampleName, "temp")
        
        logger.debug("Load instance of sample from temp dir")
        sample = pickle.load(open(os.path.join(pathPickles, 
                                               f"{sampleName}_sample.pkl"), "rb"))
        
        # Check if model version is still the same
        logger.debug("Init seg/classif models")
        if not sample.segmenter == self.seg.pathModel:
            logger.debug("Continue with different seg model")
            # Laden des ursprünglichen Models könnte noch implementiert werden
            

        if not sample.classifier == self.classif.pathModel:
            logger.debug("Continue with different classif model")
            # Laden des ursprünglichen Models könnte noch implementiert werden
            
        sample.segmenter = self.seg
        sample.classifier = self.classif
            
        # Load instances of evaluated image regions
        logger.debug("Load img regions from temp folder")
        sample.lstImgRegions = [pickle.load(open(os.path.join(pathPickles, 
                                                              regPkl), "rb")) 
                                for regPkl in os.listdir(os.path.join(pathPickles)) 
                                if not regPkl.endswith("_sample.pkl")]
        
        return sample 
        
    
    def addToEvalList(self, sample: evaluation.Evaluator):
        with open('evaluatedSamples.txt', 'a', encoding="utf-8") as fd:
            fd.write("\n" + sample.nameSample+"nothing")
            
    
        
    def initSeg(self, path = None):
        if not path:
            path = self.config.get("SEG", "ModelSegPath")
        
        # Initialize segmenter
        logger.debug("Load segmenter config")
        segConfig = segmenter.PollenConfig()

        logger.debug("Set segmenter config to values from config.ini")
        segConfig.DETECTION_MIN_CONFIDENCE = self.config.getfloat("SEG", "SegmentTresh")

        logger.debug("Get seg class names from config.ini")
        classNames = self.config.get("SEG", "ClassNames").split(",")

        logger.info("Initialize segmenter")
        return segmenter.PomoSegmentation(path, segConfig, classNames)
        
        
    def initClassif(self, path = None):
        if not path:
            path = self.config.get("CLASSIF", "ModelClassifPath")
            
        # Initialize classifier
        logger.info("Initialize classifier")
        pomoClassifier = classifier.PomoClassification(path)

        pathPollenTreshold = os.path.join("src/config", 
                                          "tresholdPollen_" + 
                                          f"{pomoClassifier._modelVersion}" + 
                                          ".ini")
        
        
        pollenTreshold = self.loadPollenTreshold(pathPollenTreshold)

        pomoClassifier.setSpeciesNames(pollenTreshold)
        return pomoClassifier
        
      