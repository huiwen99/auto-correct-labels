FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        ffmpeg

RUN conda create -n ffcv python=3.9 \
        cupy \
        pkg-config \
        compilers \
        libjpeg-turbo \
        opencv \
        pytorch \
        torchvision \
        albumentations \
        pandas \
        cudatoolkit=11.3 \
        numba -c pytorch -c conda-forge 

RUN echo "source activate" >> ~/.bashrc
RUN echo "conda activate ffcv" >> ~/.bashrc

RUN git clone https://github.com/libffcv/ffcv.git  

RUN conda run -n ffcv pip install ffcv


# docker build -t auto-correct-labels docker