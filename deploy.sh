#!/bin/bash

PORT="/dev/ttyUSB0"

PID=$(lsof -t $PORT)
[ -n "$PID" ] && sudo kill -9 $PID
sudo chmod 666 $PORT

/home/streamscope/StreamScopeDeploy/venv/bin/python /home/streamscope/StreamScopeDeploy/src/deploy.py
