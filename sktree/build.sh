cp *.py *.pyx *.pxd ../scikit-learn/sklearn/tree
cd ../scikit-learn
$CONDA_PREFIX/bin/pip install --verbose --no-build-isolation --editable .