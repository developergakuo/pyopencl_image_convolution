
import numpy as np
from scipy.misc import imread, imsave
import math
import os
import PIL.Image
import cv2
import pyopencl as cl
import pprint
import helper_functions as hf # import sequential implementation for validation
import sys
import imageio
import time
from datetime import datetime, timedelta
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#===============================================================================================================================
# SEECTION:  BENCHMARK  ARGUMENTS

image_path = sys.argv[1]
local_size = int(sys.argv[2])
kernel_version =  int(sys.argv[3])
iterations_count = int(sys.argv[4])
kernel_dim  = int(sys.argv[5])
platform  = int(sys.argv[6])

#===============================================================================================================================
# SEECTION:  HELPER FUNCTIONS

# Cconvert image to rgba for easier float4 processing and add padding
def process_image(path_to_image, padding):
    rgb_image = PIL.Image.open(path_to_image)
    rgba_image = rgb_image.convert('RGBA')
    im_src = np.array(rgba_image).astype(dtype=np.float32)
    BLANK = [0,0,0,0]
    im_src2= cv2.copyMakeBorder(im_src,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=BLANK)
    return im_src2.astype(dtype=np.float32)

def build_kernel_program(kernel_version):
  if kernel_version==1:
    return cl.Program(ctx, srcLocal).build()
  elif kernel_version ==2:
    return cl.Program(ctx, srcVector4).build()
  elif kernel_version == 3:
    return cl.Program(ctx, srcConstant).build()
  elif kernel_version == 4:
    return cl.Program(ctx, srcGlobal).build()




# Kernel parameters

kernel_sig = 1
convolution_kernel = hf.gaussian_kernel(kernel_dim, kernel_sig)  # gaussian_kernel(kernel_dim, kernel_sig)


#===============================================================================================================================
# SEECTION:  THE CONVOLUTION KERNEL SECTION

half_kernel_size = (int)(kernel_dim/2)

#Vectorized kernel with local storage and filter in constant storage
srcLocal = """
__kernel void convolute(
  const __global float4 * input, 
  __global float4 * output,
  __constant float4 * filter,
  int height, 
  int width,
  int kernelSize)
  {{

const int HALF_FILTER_SIZE = (int)(kernelSize/2);
//**********************************************************************************
    // local storage section
    
    int i = get_group_id(0);
    int j = get_group_id(1); //Identification of work-item
    int workGroupSize = {local_size};
    int idX = get_local_id(0);
    int idY = get_local_id(1);
    int ii = i*workGroupSize + idX; // == get_global_id(0);
    int jj = j*workGroupSize + idY; // == get_global_id(1);
    
    printf("%d",ii);

  __local float4 P[{local_size}+{half_kernel_size}*2][{local_size}+{half_kernel_size}*2]; // local stororage
   
    //Read pixels into local storage
    P[idX][idY] = input[ii * width + jj];

    printf("%f",input[ii * width + jj]);

    //read the rest of pixels that will lie outside the group-area but will be covered by the filter 
    if (idX < HALF_FILTER_SIZE ){{
       // 1. left side of the local area
       ii = ii -  HALF_FILTER_SIZE;
       idX =  idX - HALF_FILTER_SIZE;

       P[idX][idY] = input[ ii * width + jj];

      // 2. right side of the local area
       ii = ii +  workGroupSize; 
       idX =  idX + workGroupSize;
       P[idX][idY] = input[ ii * width + jj];

    }}
      
     if (idY < HALF_FILTER_SIZE ){{
        
       // 1. top side of the local area
       jj = jj -  HALF_FILTER_SIZE; 
       idY =  idY - HALF_FILTER_SIZE;

       P[idX][idY] = input[ ii * width + jj];
       
       // 2. bottom side of the local area

       jj = jj +  workGroupSize; 
       idY =  idY + workGroupSize;
       P[idX][idY] = input[ii * width + jj];

       

    }}
     //make sure all pixels are in local storage before you start the covolution computations
     barrier(CLK_LOCAL_MEM_FENCE);

//**********************************************************************************
      //compute convolution

    int fIndex = 0;
    float4 sum = (float4) 0.0;

        
    for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
    {{
      idX+=r;
      for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
      {{       
        idY+=c;
        sum += P[idX][idY] * filter[ fIndex ];
        fIndex++;
      }}
    }}

    barrier(CLK_LOCAL_MEM_FENCE);
    output[ii * width + jj] = sum;
}}
""".format(local_size=local_size, half_kernel_size= half_kernel_size)

#Vectorized kernel with filter in constant storage
srcVector4 = '''
__kernel void convolute(
  const __global float4 * input, 
  __global float4 * output,
  __constant float4 * filter,
  int height, 
  int width,
  int kernelSize)
  {{
    const int HALF_FILTER_SIZE = (int)(kernelSize/2);
    int ii = get_global_id(0) - {local_size};
    int jj = get_global_id(1) -{local_size};
  
    int fIndex = 0;
    float4 sum = (float4) 0.0;   
    for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
    {{
      ii+=r;
      for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
      {{      
        jj+=c;
        sum += input[ ii * width + jj] * filter[ fIndex ];
        fIndex++;
      }}
    }}

    barrier(CLK_LOCAL_MEM_FENCE);
    output[ii * width + jj] = sum;
}}
'''.format(local_size=local_size)

#kernel with image input, output, and kernel/filter in gloabal memory
srcGlobal ='''
    __kernel void convolute(
  const __global float * input, 
  __global float * output,
  __global float * filter,
  int height, 
  int width,
  int kernelSize
)
{{
  int x = get_global_id(0) - {local_size};
  int y = get_global_id(1) - {local_size} ;
  int rowOffset =x * width * 4;
  int my = 4 * y + rowOffset;
  int HALF_FILTER_SIZE = (int)({kernelSize}/2);

  int fIndex = 0;
  float sumR = 0.0;
  float sumG = 0.0;
  float sumB = 0.0;
  float sumA = 0.0;
  
    
  for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
  {{
    int curRow = my + r * (width * 4);
    for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex += 4)
    {{
      int offset = c * 4;
        
      sumR += input[ curRow + offset   ] * filter[ fIndex   ]; 
      sumG += input[ curRow + offset+1 ] * filter[ fIndex+1 ];
      sumB += input[ curRow + offset+2 ] * filter[ fIndex+2 ]; 
      sumA += input[ curRow + offset+3 ] * filter[ fIndex+3 ];
    }}
  }}
  
  output[ my     ] = sumR;
  output[ my + 1 ] = sumG;
  output[ my + 2 ] = sumB;
  output[ my + 3 ] = sumA;
  
}}
'''.format(local_size=local_size,kernelSize=kernel_dim)

#kernel with filter in constant memory
srcConstant ='''
    __kernel void convolute(
  const __global float * input, 
  __global float * output,
  __constant float * filter,
  int height, 
  int width,
  int kernelSize
)
{{
  int x = get_global_id(0) - {local_size};
  int y = get_global_id(1) - {local_size} ;
  int rowOffset =x * width * 4;
  int my = 4 * y + rowOffset;
  int HALF_FILTER_SIZE = (int)(kernelSize/2);

  int fIndex = 0;
  float sumR = 0.0;
  float sumG = 0.0;
  float sumB = 0.0;
  float sumA = 0.0;
  
    
  for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
  {{
    int curRow = my + r * (width * 4);
    for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex += 4)
    {{
      int offset = c * 4;
        
      sumR += input[ curRow + offset   ] * filter[ fIndex   ]; 
      sumG += input[ curRow + offset+1 ] * filter[ fIndex+1 ];
      sumB += input[ curRow + offset+2 ] * filter[ fIndex+2 ]; 
      sumA += input[ curRow + offset+3 ] * filter[ fIndex+3 ];
    }}
  }}
  
  output[ my     ] = sumR;
  output[ my + 1 ] = sumG;
  output[ my + 2 ] = sumB;
  output[ my + 3 ] = sumA;
  
}}
'''.format(local_size=local_size)

#===============================================================================================================================
# SEECTION:  GPU SETUP

platforms = cl.get_platforms()

print ("\nNumber of OpenCL platforms:", len(platforms))

print ("\n-------------------------")

# Investigate each platform
i =0 
for p in platforms:
    # Print out some information about the platforms
    print("plat", i)
    print ("Platform:", p.name)
    print ("Vendor:", p.vendor)
    print ("Version:", p.version)
    
    # Discover all devices
    devices = p.get_devices()
    print ("Number of devices:", len(devices))

    # Investigate each device
    i+=1
    j=0
    for d in devices:
        print( "\t-------------------------")
        # Print out some information about the devices
        print("device", j)
        j+=1
        print ("\t\tName:", d.name)
        print("\t\tVersion:", d.opencl_c_version)
        print ("\t\tMax. Compute Units:", d.max_compute_units)
        print ("\t\tLocal Memory Size:", d.local_mem_size / 1024, "KB")
        print ("\t\tGlobal Memory Size:", d.global_mem_size / (1024 * 1024), "MB")
        print ("\t\tMax Alloc Size:", d.max_mem_alloc_size / (1024 * 1024), "MB")
        print ("\t\tMax Work-group Total Size:", d.max_work_group_size)
        print ("\t\tCache Size:", d.global_mem_cacheline_size)

        # Find the maximum dimensions of the work-groups
        dim = d.max_work_item_sizes
        print ("\t\tMax Work-group Dims:(", dim[0], " ".join(map(str, dim[1:])), ")")

        print ("\t-------------------------")

    print ("\n-------------------------")
# Get platforms, both CPU and GPU
platforms = cl.get_platforms()
p = platforms[platform]
print ("\t\tName2:", p.name)
ctx = cl.Context(p.get_devices())


# Create queue for each kernel execution
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

#Kernel function instantiation
prg = build_kernel_program(kernel_version)





