FROM nvidia/cuda:10.2-base
#FROM python:3.7
FROM continuumio/anaconda3

WORKDIR /app

# Copy the following from local drive
COPY pretrain_tf.yml .

# Need to download and install packages
#RUN pip install conda
RUN conda env create -f pretrain_tf.yml
RUN conda init
RUN /bin/bash -c activate pretrain_tf.yml

COPY run_scripts.sh .
RUN mkdir tmp
RUN mkdir models

# Run our python script/application in the Docker container.
CMD ./run_scripts.sh
