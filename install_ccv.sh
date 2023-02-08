#!/bin/bash

source ./load_ccv_modules.sh

GitRepoRoot="https://github.com/rdarie/"
RepoList=(\
"python-neo" \ # pip install neo[nixio,neomatlabio]
"ephyviewer" \
"elephant" \
"pyglmnet" \
"analysis-tools" \
"rcsanalysis" \
"peakutils" \
"umap" \
"kCSD-python" \
"scaleogram" \
"pywt" \
"scikit-lego"
)

echo "Please wait for conda to install the environment..."
conda env create -f environment.yml -v -v

echo "Please check if installation was successful"
read FILLER
chmod +x $HOME/anaconda/isi_analysis/bin/*
conda activate isi_analysis
cd ..

export PYTHONPATH="/users/rdarie/anaconda/isi_analysis/lib/python3.8/site-packages"

# jupyter requires the qt console, installing after the fact to ensure the proper version
conda install jupyter --freeze-installed
conda install pyqtgraph=0.10.0 --freeze-installed
conda install -c conda-forge pyerfa --freeze-installed
conda install -c conda-forge astropy --freeze-installed

pip install pyqt5==5.10.1 --target=$PYTHONPATH --upgrade
pip install neo[nixio,neomatlabio] --target=$PYTHONPATH --no-deps
pip install vg==1.6.1 --target=$PYTHONPATH --no-deps

# pip install git+https://github.com/hector-sab/ttictoc@v0.4.1 --target=$PYTHONPATH --no-deps
pip install git+https://github.com/raphaelvallat/pingouin@v0.5.3 --target=$PYTHONPATH --no-deps
pip install git+https://github.com/melizalab/libtfr --target=$PYTHONPATH --no-deps

# WIP
# MATLABROOT="/gpfs/runtime/opt/matlab/R2021a"
# PACKAGE_DIR=$(pwd)
# cd "${MATLABROOT}/extern/engines/python"
# python setup.py build --build-base="/users/rdarie/matlab_engine_temp" install

for i in ${RepoList[*]}; do
    echo $GitRepoRoot$i".git"
    git clone $GitRepoRoot$i".git"
    cd $i
    # git checkout tags/ndav0.3
    python setup.py develop --install-dir=$PYTHONPATH --no-deps
    cd ..
done

cd ISI_Analysis_Python
python setup.py develop --install-dir=$PYTHONPATH --no-deps
