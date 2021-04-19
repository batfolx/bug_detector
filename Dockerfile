FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y sudo
RUN sudo apt-get install -y python3-dev
RUN sudo apt-get install -y python3-pip
RUN sudo pip3 install tensorflow
RUN sudo pip3 install matplotlib
RUN sudo pip3 install tf_slim
RUN sudo pip3 install scipy
RUN sudo pip3 install tf-models-official
RUN mkdir /home/detector
WORKDIR /home/detector
COPY . .
RUN export PYTHONPATH=$PYTHONPATH:tensor/models/
RUN export PYTHONPATH=$PYTHONPATH:tensor/models/official
RUN export PYTHONPATH=$PYTHONPATH:tensor/model/research/slim

