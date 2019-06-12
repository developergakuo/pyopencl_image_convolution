
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
    int ii = (i*workGroupSize + idX); // == get_global_id(0);
    int jj = (j*workGroupSize + idY); // == get_global_id(1);
    
    //printf("(%d)",HALF_FILTER_SIZE);

  __local float4 P[{local_size}+{half_kernel_size}*2][{local_size}+{half_kernel_size}*2]; // local stororage
   
    //Read pixels into local storage
    //P[idX][idY] = input[ii * width + jj];
     //printf("%f",input[ii * width + jj]);

    //read the rest of pixels that will lie outside the group-area but will be covered by the filter 
    if (idX < HALF_FILTER_SIZE ){{
       // 1. left side of the local area
       int X = ii -  HALF_FILTER_SIZE;
       int x =  idX - HALF_FILTER_SIZE;
       //printf(" -x-> (%d-%d)",ii,X);
       P[x][idY] = input[ X * width + jj];

      // 2. right side of the local area
       int X2 = ii +  workGroupSize; 
       int x2 =  idX + workGroupSize;
       //printf(" +x2->(%d-%d)",ii,X2);
       P[x2][idY] = input[ X2 * width + jj];

    }}
      
     if (idY < HALF_FILTER_SIZE ){{
    
       // 1. top side of the local area
        int Y = jj -  HALF_FILTER_SIZE; 
       int y =  idY - HALF_FILTER_SIZE;
       //print(" -y->(%d-%d)",jj,Y);
       P[idX][y] = input[ ii * width + Y];
       
       // 2. bottom side of the local area

       int Y2= jj +  workGroupSize; 
       int y2 =  idY + workGroupSize;
       //printf(" +y2->(%d-%d)",jj,Y2);
       P[idX][y2] = input[ii * width + Y2];

       

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
        //sum += P[idX][idY] * filter[ fIndex ];
        fIndex++;
      }}
    }}
    
    barrier(CLK_LOCAL_MEM_FENCE);
    output[ii * width + jj] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
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


# Get platforms, both CPU and GPU
platforms = cl.get_platforms()
p = platforms[platform]
ctx = cl.Context(p.get_devices())


# Create queue for each kernel execution
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

#Kernel function instantiation
prg = build_kernel_program(kernel_version)

#Read in image
im_dir = os.path.split(os.path.realpath(__file__))[0]
image_padding = local_size
im_src = process_image(image_path,image_padding)
(width_g,height_g,depth)=im_src.shape
src_buff = cl.image_from_array(ctx, im_src, mode='r')
#Allocate memory for variables on the device

d_kernel = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=convolution_kernel)
img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=im_src)
result_g = cl.Buffer(ctx, mf.WRITE_ONLY, im_src.nbytes)
# Call Kernel. Automatically takes care of block/grid distribution
local_size = (local_size,local_size)
global_size = (height_g,width_g)

#===============================================================================================================================
# SEECTION:  RUN PARALLEL BENCHMARKS
for i in range(iterations_count):
    x = 10
    gpu_start_time = datetime.now()
    while x > 0:
        prg.convolute(queue, global_size, local_size ,
         img_g, result_g,d_kernel,np.int32(height_g),np.int32(width_g),np.int32(kernel_dim),
         global_offset=[image_padding,  image_padding])
        result = np.empty_like(im_src)
        cl.enqueue_copy(queue, result, result_g)
        queue.finish()
        x -= 1
    gpu_end_time = datetime.now()
    print((gpu_end_time - gpu_start_time).total_seconds())
#===============================================================================================================================
# SEECTION:  RESHAPE OUTPUT AND VALIDATE SOLUTION
def reshape_result_and_validate(image_path,result,height_g,width_g, image_padding):
  tolerance =3.00000000e+02
  #shape output solution
  half_height = (int)(height_g/2)
  result2 = result[:,half_height:height_g,:3]
  result1 = result[:,:half_height,:3]
  result = np.concatenate((result2,result1),axis = 1)[image_padding:width_g-image_padding,image_padding*2:height_g,:3]
  #save image
  imageio.imwrite('medianFilter-OpenCL1.png',result)

  #validate solution
  img_arr = hf.image_to_array(image_path)
  (height_g, width_g, depth) = img_arr.shape
  img_src = img_arr.reshape((height_g * width_g, 3))
  expected = hf.apply_kernel_1d(convolution_kernel, kernel_dim, height_g, width_g, img_src)
  result = result.reshape((height_g * width_g, 3))
  print("Result equals expected?", np.array_equal(expected, result))

  print("equal within tolerance?",np.allclose(expected, result, rtol=tolerance, atol=tolerance, equal_nan=True))

#reshape_result_and_validate(image_path,result,height_g,width_g, image_padding)




