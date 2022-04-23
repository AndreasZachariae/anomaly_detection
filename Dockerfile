FROM tensorflow/tensorflow:2.3.0-gpu

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-tk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install pandas matplotlib

CMD /bin/bash