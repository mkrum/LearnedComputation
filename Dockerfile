FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /root/

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/mkrum/LearnedComputation.git lc && \
            cd lc && \
            pip install -r requirements.txt

WORKDIR /root/lc
