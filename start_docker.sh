docker build -t tensorflow-gpu:2.3.0 .

xhost + local:root
docker run --gpus all --name tensorflow-gpu -v $(pwd):/anomaly_detection -v /tmp/.X11-unix:/tmp/.X11-unix -it --privileged -e DISPLAY=$DISPLAY --rm tensorflow-gpu:2.3.0