# -*- coding: utf-8 -*-

import helper_functions as hf
from datetime import datetime, timedelta
import sys




#########
# Input #
#########

img_arr = hf.image_to_array(sys.argv[1])
iterations = int(sys.argv[2])
kernel_dim = int(sys.argv[3])

(img_h, img_w, bytes_per_pixel) = img_arr.shape

##############
# Sequential #
##############

# Kernel parameters
kernel_sig = 1
kernel = hf.gaussian_kernel(kernel_dim, kernel_sig)  # gaussian_kernel(kernel_dim, kernel_sig)


# Image input array
img_src = img_arr
img_src = img_src.reshape((img_h * img_w, 3)) # flat list of lists
for i in range(int(iterations)):
    x = 10
    start_time = datetime.now()
    while x > 0:
        img_dst = hf.apply_kernel_1d(kernel, kernel_dim, img_h, img_w, img_src)
        x -= 1
    end_time = datetime.now()
    print((end_time - start_time).total_seconds())    
    img_dst = img_dst.reshape((img_h, img_w, bytes_per_pixel))



# Write the image to a file
#save_image(img_dst, "output_seq.jpg")
