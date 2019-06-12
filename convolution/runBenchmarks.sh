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



for (( u = 1 ; u <= 4; u += 1))
do
        echo "---"
        echo "> Parallel kernel_version tests)"
        echo "> GPU"
        python  parallel.py $image_path $local_size2 $u $iterations $kernel_size $platform > out-par-kernels_version$u-$platform.csv
        echo "> CPU"
        python  parallel.py $image_path $local_size2 $u $iterations $kernel_size $platform1 > out-par-kernels_version$u-$platform1 .csv

        echo "---"
done

for (( u = 3 ; u <= 15; u += 2))
do
        echo "---"
        echo "> Parallel kernel_size tests)"
        echo "> GPU"
        python  parallel.py $image_path $local_size2 $kernel_version $iterations1 $u $platform  > out-par-kernels_size$u-$platform.csv
        echo "> CPU"
        python  parallel.py $image_path $local_size2 $kernel_version $iterations1 $u $platform1  > out-par-kernels_size$u-$platform1.csv
        echo "---"

      
done

for (( u = 8 ; u <= 32; u += 8))
do
        echo "---"
        echo "> Parallel local_size tests)"
        echo "> GPU"
         python  parallel.py $image_path $u $kernel_version $iterations $kernel_size $platform > out-par-local_size$u-$platform.csv
        echo "> CPU"
        python  parallel.py $image_path $u $kernel_version $iterations $kernel_size $platform1 > out-par-local_size$u-$platform1.csv
        echo "---"
done


for (( u = 3 ; u <= 11; u += 2))
do
        for (( v = 1 ; v <= 4; v += 1))
        do
                echo "---"
                echo "> Parallel kernel_version tests)"
                echo "> GPU"
                python  parallel.py $image_path $local_size2 $v $iterations $u $platform > out-par-kernels_version-load$u-$v-$platform .csv
                echo "> CPU"
                python  parallel.py $image_path $local_size2 $v $iterations $u $platform1 > out-par-kernels_version-load$u-$v-$platform1 .csv
                echo "---"
        done     
done