FROM nvidia/cuda:10.2-base
FROM continuumio/miniconda3

WORKDIR /app

# Copy the following from local drive
COPY pretrain_tf.yml .
RUN conda env create -f pretrain_tf.yml python=3.7

COPY codes codes/
COPY pt_medium.txt .
COPY run_scripts.sh .

RUN mkdir tmp
RUN mkdir models

# Run our python script/application in the Docker container.
CMD conda run -n pretrain_tf ./run_scripts.sh 

# to build:  $docker image build -t name:latest .
# to run:  $docker run -it --mount src="$(pwd)",target=/app/models,type=bind pretrained:latest