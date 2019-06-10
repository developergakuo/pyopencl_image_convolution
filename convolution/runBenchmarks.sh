#!/usr/bin/env bash

local_size1=16
local_size2=8
iterations=10
image_path='inputs/sonic.jpg'
kernel_size=5
kernel_version=1



for (( u = 1 ; u <= 4; u += 1))
do
        echo "---"
        echo "> Parallel kernel_version tests)"
        python  parallel.py $image_path $local_size2 $u $iterations $kernel_size > out-par-kernels_version$u.csv
        echo "---"
done

for (( u = 3 ; u <= 7; u += 2))
do
        echo "---"
        echo "> Parallel kernel_size tests)"
        python  parallel.py $image_path $local_size2 $kernel_version $iterations $u > out-par-kernels_size$u.csv
        echo "---"

        echo "---"
        echo "> sequential kernel_size tests)"
        python3  sequential.py $image_path $iterations $u > out-seq-kernels_size$u.csv
        echo "---"
done

for (( u = 8 ; u <= 16; u += 8))
do
        echo "---"
        echo "> Parallel local_size tests)"
        python  parallel.py $image_path $u $kernel_version $iterations $iterations > out-par-local_size$u.csv
        echo "---"
done