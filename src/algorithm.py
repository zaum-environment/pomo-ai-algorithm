# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:00:36 2022
@author: stemmler_t (Tom Stemmler, t.stemmler@hund.de)
"""

version = "v1.34.0.1"



# # Lizenzinformationen in deutsch, License information in english below

# Dieses Projekt verwendet das VGG19-Modell und das Mask R-CNN-Modell unter den folgenden Lizenzbedingungen:

# ## VGG19-Modell
# Das VGG19-Modell wurde ursprünglich von der Visual Geometry Group an der Universität Oxford entwickelt. Das Modell wurde über [TensorFlow/Keras] bezogen und unterliegt den Bedingungen der Apache 2.0 Lizenz (für TensorFlow/Keras).

# - TensorFlow: [Apache 2.0 Lizenz](https://www.tensorflow.org/license)
# - Keras: [Apache 2.0 Lizenz](https://github.com/keras-team/keras/blob/master/LICENSE)

# ## Mask R-CNN-Modell
# Das Mask R-CNN-Modell wurde von Facebook AI Research entwickelt. Das Modell wurde über [TensorFlow/Keras] bezogen und unterliegt den Bedingungen der Apache 2.0 Lizenz.

# - TensorFlow: [Apache 2.0 Lizenz](https://www.tensorflow.org/license)
# - Keras: [Apache 2.0 Lizenz](https://github.com/keras-team/keras/blob/master/LICENSE)


# ## Eigener Code
# Der restliche Code in diesem Projekt ist Eigentum der Firma "Helmut Hund GmbH, Artur-Herzog-Straße 2, 35580 Wetzlar" und entwickelt von "Tom Stemmler, t.stemmler@hund.de" und ist unter der [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) Lizenz](https://creativecommons.org/licenses/by-nc/4.0/) lizenziert. Dies bedeutet, dass Sie den Code für nicht-kommerzielle Zwecke nutzen dürfen, solange Sie die Quelle angeben.

# ```plaintext
# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

# Sie dürfen:
# - Teilen — das Material in jedwedem Format oder Medium vervielfältigen und weiterverbreiten
# - Bearbeiten — das Material remixen, verändern und darauf aufbauen

# Unter den folgenden Bedingungen:
# - Namensnennung — Sie müssen angemessene Urheber- und Rechteangaben machen, einen Link zur Lizenz beifügen und angeben, ob Änderungen vorgenommen wurden. Diese Angaben dürfen in jeder angemessenen Art und Weise gemacht werden, allerdings nicht so, dass der Eindruck entsteht, der Lizenzgeber unterstütze gerade Sie oder Ihre Nutzung besonders.
# - Nicht kommerziell — Sie dürfen das Material nicht für kommerzielle Zwecke nutzen.

# Es gelten keine weiteren Einschränkungen — Sie dürfen keine zusätzlichen Klauseln oder technische Verfahren einsetzen, die anderen rechtlich irgendetwas untersagen, was die Lizenz erlaubt.





# # License Information

# This project uses the VGG19 model and the Mask R-CNN model under the following licenses:

# ## VGG19 Model
# The VGG19 model was originally developed by the Visual Geometry Group at the University of Oxford. The model was obtained via [TensorFlow/Keras] and is subject to the terms of the Apache 2.0 License (for TensorFlow/Keras).

# - TensorFlow: [Apache 2.0 License](https://www.tensorflow.org/license)
# - Keras: [Apache 2.0 License](https://github.com/keras-team/keras/blob/master/LICENSE)

# ## Mask R-CNN Model
# The Mask R-CNN model was developed by Facebook AI Research. The model was obtained via [TensorFlow/Keras] and is subject to the terms of the Apache 2.0 License.

# - TensorFlow: [Apache 2.0 License](https://www.tensorflow.org/license)
# - Keras: [Apache 2.0 License](https://github.com/keras-team/keras/blob/master/LICENSE)


# ## Custom Code
# The remaining code in this project is owned by Company "Helmut Hund GmbH, Artur-Herzog-Straße 2, 35580 Wetzlar" and developed by "Tom Stemmler, t.stemmler@hund.de", and is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. This means you are free to use the code for non-commercial purposes, with appropriate attribution.

# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

# You are free to:
# - Share — copy and redistribute the material in any medium or format
# - Adapt — remix, transform, and build upon the material

# Under the following terms:
# - Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial — You may not use the material for commercial purposes.

# No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

# This is a human-readable summary of (and not a substitute for) the license. The full license text can be found at:

# https://creativecommons.org/licenses/by-nc/4.0/legalcode





"""
--------------------------------------
IMPORT LIBS
--------------------------------------

"""

# setup fist logger befor import other modules. Otherwise the logging will not
# work correctly.
from libs.pomoLib import pomoUtils
from datetime import datetime
import time
import os



now = datetime.now()
dateTime = now.strftime("%Y%m%d-%H%M%S")
logger = pomoUtils.setup_logger("root", f"logs/{dateTime}_PomoAI.log", debug = False)

logger.info("Load libs")

from configparser import ConfigParser

from libs.pomoLib import app



if __name__ == "__main__":  
    # try:
    """
    --------------------------------------
    INITIALIZATION
    --------------------------------------
    
    """
    
    
    # Get config
    logger.debug("Load config file") 
    # Load config parameter from config fie
    config = ConfigParser()
    # Load original key, not lower case
    config.optionxform = str
    config.read("src/config/config.ini")
    #config.read("config/config.ini")
    #Create instance of PomoAI and do the initalisation
    logger.debug("Create a instance of PomoAI application")
    PomoAI = app.PomoAI(config,version)
    
    flagMsgOut = True
    
    
    #%%
    
    """
    --------------------------------------
    MAINLOOP
    --------------------------------------
    
    """
    logger.info("Starting Mainloop")

    # Check for new samples to evaluate
    sampleInfo = PomoAI.checkForNewSample()

    while sampleInfo:
        start = time.time()
        
        # create new sample instance
        PomoAI.createNewSampleEvaluator(sampleInfo)
        # set flag for msg output to True (Waiting for new sample)
        flagMsgOut = True
    
        active = False
        for sample in PomoAI.lstOpenSamples:
            while sample.active:
                active = True
                imgRegion = sample.nextImageRegion()
                if imgRegion:
                    sample.analyzeRegion(imgRegion)
                    PomoAI.saveSamplePickle(sample)
                else:
                    if sample.endOfSample:
                        sample.sampleEnd()
                        PomoAI.addToEvalList(sample)
                        
                        PomoAI.lstOpenSamples.remove(sample)
                    else:
                        logger.info(f"Sample {sample.nameSample} set to inactive")
                        logger.info("Waiting for new sample or new img region")
                        sample.active = False
            
        end = time.time()
        print("Processing time: ", end - start)

        sampleInfo = PomoAI.checkForNewSample()
    
        if not PomoAI.lstOpenSamples:
            if flagMsgOut:
                logger.info("Waiting for new sample")
                flagMsgOut = False
                
            time.sleep(config.getint('MAIN', 'SleepingTime'))
        elif PomoAI.lstOpenSamples:
            time.sleep(2)
                    
        #except Exception as e:
            #logger.error(e)


            

