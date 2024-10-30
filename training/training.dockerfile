# Dockerfile for model training.

FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install ktrain && \
    pip3 install pandas && \
    pip3 install -q pyyaml h5py 

#RUN pip3 uninstall numpy 

RUN pip3 install tf_keras
RUN pip3 install numpy==1.24.0     
ADD training.py /
ADD training.yml /

ENV TF_USE_LEGACY_KERAS=True

