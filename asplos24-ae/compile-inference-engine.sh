#!/bin/sh

cd ../inference-engine
mkdir -p build
cd build
cmake ..
make -j
