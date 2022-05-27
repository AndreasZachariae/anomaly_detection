FROM tensorflow/tensorflow:2.3.0-gpu

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-tk \
    ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install \
    pandas \
    matplotlib \
    opencv-contrib-python \
    scikit-learn

CMD /bin/bash