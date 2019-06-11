#!/usr/bin/env bash

local_size1=16
local_size2=8
iterations=30
iterations1=3
image_path='inputs/sonic.jpg'
kernel_size=5
kernel_version=1
platform=1
platform1=0





for (( u = 3 ; u <= 11; u += 2))
do
        for (( v = 1 ; v <= 4; v += 1))
        do
                echo "---"
                echo "> Parallel kernel_version tests)"
                echo "> GPU"
                echo "> CPU"
                python  parallel.py $image_path $local_size2 $v $iterations $u $platform1 > out-par-kernels_version-load$u-$v-$platform1 .csv
                echo "---"
        done     
done



