FROM ucsdets/scipy-ml-notebook
LABEL maintainer="Yuxuan Fan <yufan@ucsd.edu>"
USER root
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN pip install --no-cache-dir gensim pandas-profiling Pillow