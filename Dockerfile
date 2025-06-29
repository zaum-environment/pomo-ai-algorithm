# This is the Dockerfile a algorithm developer has to provide. It will be used by SYLVA IT Infrastructure 
# to build an image that will be used to run the algorithm.

# Define whatever base image is fine for your algorithm. The only requirement is that it provides the 
# possibility to call a script "runAlgorithm" that will be executed by SYLVA IT Infrastructure. Linux 
# based images are recommended.
# FROM ubuntu:latest

# Prepare the container for the algorithm. This includes installing all necessary packages and setting 
# up the environment.
FROM python:3.8
RUN apt-get update && apt-get install  ffmpeg libsm6 libxext6 unzip -y && \
 pip install h5py opencv-python==4.7.0.72 pandas parso pillow pyzmq scikit-image==0.19.3 scikit-learn==1.3.2 scipy seaborn tensorflow==2.4.0 zipp numpy==1.19.5 imageio imgaug keras-preprocessing==1.1.2 lxml IPython gdown && \
 # As the algorithm will be executed by a non-root user, the output folder has to be writable for all
 mkdir -p /data/output && chmod a+rx /data && chmod a+rwx /data/output

# The startAlgorithm script is the entry point of the container. It is the script that will be executed 
# when the container is run by SYLVA IT infrastructure. The script itself should just start your algorithm 
# to get all files in folder /data/input processed.

# Any working directory can be chosen as per choice like '/' or '/home' etc
WORKDIR /wd

#to COPY the remote file at working directory in container
RUN mkdir src
COPY src ./src/

COPY src/bin/startAlgorithm /bin/startAlgorithm
RUN chmod a+x /bin/startAlgorithm

RUN unzip src/models/models.zip -d src/

RUN mkdir -p src/logs
RUN chmod -R a+rwx /wd

#CMD ["python", "src/algorithm.py"]
