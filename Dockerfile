FROM pytorch/pytorch:latest

RUN pip install cython
RUN pip install opencv-python pillow pycocotools matplotlib
RUN git clone https://github.com/HongYang9788/FBL_YOLACT.git
WORKDIR /FBL_YOLACT

ENTRYPOINT ["bash"]