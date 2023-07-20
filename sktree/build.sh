cp *.py *.pyx *.pxd ../scikit-learn/sklearn/tree
cd ../scikit-learn
pip install --verbose --no-build-isolation --editable .