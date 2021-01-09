FROM ucsdets/datascience-notebook:2020.2-stable

LABEL maintainer="Yuxuan Fan <yufan@ucsd.edu>"

USER root

RUN apt-get update && apt-get install -y g++ openjdk-8-jdk curl time gzip

RUN pip install --no-cache-dir numpy pandas scikit-learn Pillow nltk gensim

RUN echo -e "#!/usr/bin/env bash\n\njupyter notebook \"$@\"\n" > run_jupyter.sh && chmod 755 run_jupyter.sh

USER $NB_UID
