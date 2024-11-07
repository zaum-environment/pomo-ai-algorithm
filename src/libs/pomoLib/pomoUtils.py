# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:39:00 2022

@author: stemmler_t
"""

import os
import logging 
import zipfile
import tempfile

from libs.pomoLib.datatypes import pathComp

def setup_logger(logger_name,logfile, debug = True):
    """
    Setup and returns a logger. Just use once in main module. Paramters are
    the name of the logger, the name of the logfile (and path) and a flag 
    which set the logger to debug mode.

    Parameters
    ----------
    logger_name : TYPE
        DESCRIPTION.
    logfile : TYPE
        DESCRIPTION.
    debug : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    logger : TYPE
        DESCRIPTION.

    """
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
   
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    if debug:
        ch.setLevel(logging.DEBUG)
        
    # create formatter and add it to the handlers
    fhformatter = logging.Formatter('%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(message)s')
    chformatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    
    fh.setFormatter(fhformatter)
    ch.setFormatter(chformatter)
    # add the handlers to the logger
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.debug("RootLogger has been initialized")
    return logger



def getPathInfo(path:str):
    """
    This function checks if the given path string "path" is in a suitable
    format to get read by PomoAI. If so, the string contains 7 elements which
    are seperate by a "-" and giving information about:
        - imgRegionPos
        - date
        - device
        - barcode
        - type (tif, png)
        
    If string has wrong format, the function returns None

    Parameters
    ----------
    path : str
        String to be checkt and split

    Returns
    -------
    pathSplit : list
        Returns a list containing the information from the string

    """
    head, tail = os.path.split(path)
    pathSplit = tail.split("-")
    if len(pathSplit) == 7:
        return pathSplit
    elif (len(pathSplit) > 7 and pathSplit[0] == "polle" and 
          pathSplit[3] == "pmon"):
            return pathSplit
    else:
        raise Exception(f"Unexpected format of input file name({tail})")
        

def buildPath(comp: pathComp):
    #if len(comp) == 7:
    pathString = (comp[0] + "-" +
                  comp[1] + "-" +
                  comp[2] + "-" +
                  comp[3] + "-" +
                  comp[4] + "-" +   
                  comp[5] + "-" +
                  comp[6])
    
    return pathString

def modPath(path: str, pos: int, val: str):
    pathInfo = getPathInfo(path)
    if pathInfo:
        pathInfo[pos] = val
        return buildPath(pathInfo)
        
    return None
                    
                  
def unzipProbe (pathZipFile):
    if zipfile.is_zipfile(os.path.join(pathZipFile)):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(pathZipFile, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        subfolders = [f.path for f in os.scandir(temp_dir) if f.is_dir()]
        if len(subfolders) != 1:
            raise Exception("Unexpected number of subfolders")
        
        temp_dir = os.path.join(temp_dir, subfolders[0])

        return temp_dir


def getOffset(value, offset, maxVal):
    if value + offset >= value:
        if value + offset > maxVal:
            return value
    else:
        if value + offset < maxVal:
            return value
        
    return value + offset
        
        
    