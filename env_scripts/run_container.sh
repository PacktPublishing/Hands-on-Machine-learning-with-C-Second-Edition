#!/bin/bash
xhost +local:root
docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it -v $PWD/..:/samples buildenv:1.0 bash

