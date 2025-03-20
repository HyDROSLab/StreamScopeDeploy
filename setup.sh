#!/bin/bash

# Create .venv directory
pyton3 -m venv .venv

# Activate python virtual env
source .venv/bin/activate

# Install requirements
python3 -m pip install -r requirements.txt
