#!/bin/bash

# Create Env
eval "$(conda shell.bash hook)"
conda env create -f ./fareenv.yml -n fareenv
conda activate fareenv
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d/
echo $'export LD_LIBRARY_PATH_OLD=$LD_LIBRARY_PATH\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo $'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_OLD' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# Get sklearn pinned commit
wget https://github.com/scikit-learn/scikit-learn/archive/fd60379f95f5c0d3791b2f54c4d070c0aa2ac576.zip
unzip fd60379f95f5c0d3791b2f54c4d070c0aa2ac576.zip
mv scikit-learn-* scikit-learn
rm fd60379f95f5c0d3791b2f54c4d070c0aa2ac576.zip

# Patch with our sktree
cd sktree
./build.sh
cd ..
mkdir code/src/tree/out
