#!/usr/bin/env sh

echo $1

../../build/tools/caffe train \
    "--solver=/media/data3/googlenet_bn/$1/solver_stepsize_6400.prototxt" 2>&1 | tee "/media/data3/googlenet_bn/$1/googlenet_bn_stepsize6400.log"
