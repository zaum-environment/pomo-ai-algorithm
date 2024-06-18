# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:59:33 2022

@author: stemmler_t
"""

"""
--------------------------------------
Libs
--------------------------------------

"""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum



"""
--------------------------------------
Enums
--------------------------------------

"""

class sampleType(Enum):
    folder = 1
    file = 2
    zipped = 3

class pathComp(Enum):
    polle = 0
    imgRegPos = 1
    dateTime = 2
    pmon = 3
    device = 4
    barcode = 5
    elemType = 6

class imgElemType(Enum):
    tif = "tiff.tif"
    synth = "tiffSYN.png"
    seg = "tiffSEG.png"
    
    
"""
--------------------------------------
Dataclasses
--------------------------------------

"""


@dataclass
class posImg:
    xPos: str = ""
    yPos: str = ""
    zPos: str = "01"


"""
Represents a found object with all its attributes like position, segmentation 
class, classifcation class, etc.
"""
@dataclass
class dcPomoObject:
    xPos: int = field(default = 0)
    yPos: int = field(default = 0)
    zPos: int = field(default = 35)
    width: int = field(default = 0)
    height: int = field(default = 0)
    
    imgObj: list = None 
    imgObjStack: list = None
    segMask: list = field(default_factory=lambda : [])
    
    segClass: str = field(default = "")
    segScore: float = field(default = 0.0)
    
    clfSpeciesInt: int = field(default = 0)
    clfSpecies: str = field(default = "")
    clfScore: float = field(default = 0.0)
    
    clfSpeciesIntSec: int = field(default = 0)
    clfSpeciesSec: str = field(default = "")
    clfScoreSec: float = field(default = 0.0)
    
    specFolder: str = field(default = "")
    sortedOut: str = None


"""
Contains the values for the (dynamic) treshold of a specific spicies.

"num" is the number of objects per class that are effected by the dynamic
threshold. If set to 1, the dynamic treshold is not activated.

"score" represents the minimum accuracy for the classifcation. If below the 
object will be saved as "Undefind". If the dynamic treshold is activated the
minimum accuracy rate depends on the number of found objects per class and 
varies between "score" and "scoreMax".

"scoreMax" reoresents the above limit of the dynamic treshold. If the accuracy 
of a predication for all objects is above this value, the object is surely 
classified as the prediced species and the dynamic threshold will not have an 
effect for this species class.
"""
@dataclass
class dcTreshold:
    num: int
    score: int
    scoreMax: int


"""
Represents a species that is inside the trained network.
It contaions the german, latin and english name as well as the theshold values
for the recognision.
"""
@dataclass
class dcSpecies:
    nameGer: str
    nameLat: str
    nameEng: str
    subClass: str
    default: dcTreshold
    
    
    
if __name__ == "__main__":

    Hasel = dcSpecies("Hasel", "Corylus", "Hazel", "", default=dcTreshold(3, 70, 98))
    print (Hasel.default.num)






