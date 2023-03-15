# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest-gpu
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install pydot
RUN pip install scipy
RUN pip install opencv-python
RUN pip install matplotlib
WORKDIR /app
# USER lvvr
CMD python test.py

# docker run --user lvvr -it --rm --runtime=nvidia -v "$(pwd):/app" -v "$(pwd):/app" cnnps &> cnnps.log &
