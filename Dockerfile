FROM nvidia/cuda:10.2-base
FROM python:3.7

WORKDIR /app

# Copy the following from local drive
COPY requirements.txt .

# Need to download and install packages
RUN pip3 install -r requirements.txt

COPY codes codes/
COPY pt_medium.txt .
COPY run_scripts.sh .

RUN mkdir tmp
RUN mkdir models

# Run our python script/application in the Docker container.
CMD ./run_scripts.sh
