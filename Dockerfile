FROM ucsdets/scipy-ml-notebook
LABEL maintainer="Daniel Lee <dhl011@ucsd.edu> & Yuxuan Fan <yufan@ucsd.edu>"
USER root
# JDK 8
RUN apt-get update && apt-get install -y openjdk-8-jdk
# Pytorch
RUN pip uninstall --no-cache-dir -y torch torchvision && \
    pip install --no-cache-dir torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# Other Python Dependencies
RUN pip install --no-cache-dir spacy gensim pandas-profiling pandarallel sentence-transformers && \    
    python -m spacy download en_core_web_sm
