#!/bin/bash
source ./load_ccv_modules.sh
echo "Please wait for conda to install the environment..."
conda env create -f environment_octave.yml -v -v

echo "Please check if installation was successful"
read FILLER
