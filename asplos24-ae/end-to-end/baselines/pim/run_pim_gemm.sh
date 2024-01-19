#!/bin/sh

mkdir -p build
cd build
cmake ..
make -j
./bin/test_gemm ./bin/dpu_bin_gemm
