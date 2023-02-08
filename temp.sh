#!/bin/bash

source ./load_ccv_modules.sh

module unload chrome/73.0

GitRepoRoot="git://github.com/rdarie/"
RepoList=(\
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

conda activate isi_analysis
cd ..

export PYTHONPATH="/users/rdarie/anaconda/isi_analysis/lib/python3.8/site-packages"

# pip install git+git://github.com/hector-sab/ttictoc@v0.4.1 --target=$PYTHONPATH --no-deps
pip install git+git://github.com/raphaelvallat/pingouin@v0.5.3 --target=$PYTHONPATH --no-deps
pip install git+git://github.com/melizalab/libtfr --target=$PYTHONPATH --no-deps

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

