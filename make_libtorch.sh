#!/bin/bash

TARGET="./my_libtorch"

rm -rf ${TARGET}
mkdir -p ${TARGET}
cp -r /opt/conda/lib/python3.6/site-packages/torch/include ${TARGET}
cp -r /opt/conda/lib/python3.6/site-packages/torch/share ${TARGET}
cp -r /opt/conda/lib/python3.6/site-packages/torch/lib ${TARGET}
