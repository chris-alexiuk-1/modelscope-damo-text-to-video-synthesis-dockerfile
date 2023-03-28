FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.8 \
    && apt install -y python3.8-distutils \
    && rm -rf /var/lib/apt/lists/* 
    
# install cv2 dependencies     
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install python packages
COPY requirements.txt requirements.txt

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8 \
    && python3.8 -m pip install -r requirements.txt

# copy app folder
COPY ./app /app

# set working directory
WORKDIR /app

# set entrypoint
ENTRYPOINT [ "python3.8", "app.py"]
