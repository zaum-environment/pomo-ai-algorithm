# PomoAI v1.34.0.1

Hund BAA500 Algorithm.

Step 1: Scanning the sample. As the scanning routine is the most time-consuming step during the evaluation procedure, 144 XY positions across the area with the highest density of deposited particles are scanned for the most widely used 3-hour measurement interval. This results in a representative result for the sample. For a smaller number of samples per day, it is also possible to scan the full sample surface, corresponding to a total of 576 (24 x 24) XY positions. Neighbouring images within a Z stack are captured with a vertical distance of 1.5 µm. The resulting extended depth of field guarantees an entirely sharp image for all possible pollen grain sizes. The full image stack consists of 180 images with which all Z positions are covered where objects can be expected.

Step 2: Segmentation. A set of sample images is used to manually draw segmentation masks with the tool “VGG Image Annotator” (VIA). Five types of object masks are defined in this step: fungal spores, pollen, particles, fibres, and pollen fragments. The segmentation algorithm is a Region-Based convolutional neural network (Mask R-CNN). After its training, the CNN will draw a mask around any found object automatically, compare it (along with the content of the mask) with the training data and then classify the object according to the five classes mentioned above. These pre-classified objects are then cut out of the synthetic 2-D images (see above), stored in single image files and then fed into the classification network.

Step 3: Classification. After segmentation and pre-classification, single-object images are fed into the classification based on a VGG-19 CNN. This network was trained with a total of more than 150’000 single, labelled pollen objects - based on a database collected over more than 15 years of development of the BAA500.
The classification also yields a quality measure that determines the reliability of the identification process for every single object. The results are then stored in an XML output file. 

References:
- [Helmut Hund GmbH](https://www.hund.de)


## License
Please check LICENSE.txt
