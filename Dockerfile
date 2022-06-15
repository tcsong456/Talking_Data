FROM continuumio/miniconda:4.6.14

WORKDIR talking_data
COPY . .
RUN apt-get update && apt-get install -y \
    unzip \
    grep
    
RUN cd /root/ && mkdir .kaggle && cd /talking_data
RUN cp kaggle.json /root/.kaggle/kaggle.json
RUN conda env create -f talking_data_dependencies.yaml

ENTRYPOINT ["/bin/bash","-c","source activate talking-data-env && bash run.sh"]
    