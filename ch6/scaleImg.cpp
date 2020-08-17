#define _CRT_SECURE_NO_WARNINGS


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <stddef.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else  
#include <CL/cl.hpp> 
#endif

#include "imgShow.h"
#include "geterror.h"

using namespace std;
using namespace cl;

using std::size_t;

const std::string kernelCode = R"(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |  CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; 
//__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |  CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
__kernel void scaleImg(read_only image2d_t src_image,
                        write_only image2d_t dst_image) {

   /* Compute value to be subtracted from each pixel */
   float sx = 1.0f*(get_image_width(src_image)-1)/ (get_image_width(dst_image)-1);
   float sy = 1.0f*(get_image_height(src_image)-1)/(get_image_height(dst_image)-1);

   /* Read pixel value */
   int2 dstcoord = (int2)(get_global_id(0), get_global_id(1));
   float2 srccoord  = (float2)(sx*get_global_id(0), sy*get_global_id(1));
   uint4 pixel = read_imageui(src_image, sampler, srccoord);
   
   /* Write new pixel value to output */
   write_imageui(dst_image, dstcoord, pixel);
}
)";

#define KERNEL_FUNC "scaleImg"


void setupImgData(uint8_t* imgData,const int height, const int width)
{
    int blockWidth = width > 8 ? width / 8 : 1;
    int blockHeight = height > 8 ? height / 8 : 1;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            
            imgData[y * width + x] = (x)&1?0xff:0x1;
        }
    }
}
int main() 
{
    int scale = 3;
    int srcImgH = 320;
    int srcImgW = 240;
    int dstImgH = scale * srcImgH;
    int dstImgW = scale * srcImgW;
    int channel = 1;
    cl::Platform platforms = cl::Platform::getDefault();

    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, NULL);

    cl_int errNum = CL_SUCCESS;
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    /* Create a CL command queue for the device*/
    cl::CommandQueue cq = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &errNum);
    if (errNum != CL_SUCCESS) {
        printf("failed to get queue£∫%d\n", errNum);
        return -1;
    }

    /* Program/kernel data structures */
    cl::Program program = cl::Program(context, kernelCode, false, &errNum);
    errNum = program.build(devices);  //±‡“ÎProgram
    if (errNum < 0)
    {
        std::size_t log_size;
        program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG,&log_size);
        /* Find size of log and print to std output */

        char* program_log = new char[log_size + 1];
        program_log[log_size] = '\0';

        program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, program_log);        
        printf("%s\n", program_log);
        delete [] program_log;
        return -2;
    }

    cl::Kernel kernel(program, KERNEL_FUNC);
    

  
  
   
   /* Data and buffers */
   uint8_t*  srcImgData = new uint8_t[srcImgH*srcImgW*channel];
   uint8_t*  dstImgData = new uint8_t[dstImgH*dstImgW*channel];
   //memset(srcImgData, 0xff, srcImgH*srcImgW*channel*sizeof(uint8_t));
   setupImgData(srcImgData, srcImgW, srcImgH);
   //showImage(srcImgData, srcImgW, srcImgH,channel);


   cl_int err;   
   
   ImageFormat png_format(CL_R, CL_UNSIGNED_INT8);
   if (channel == 1)
   {
       png_format = ImageFormat(CL_R, CL_UNSIGNED_INT8);
   }
   else if (channel == 3)
   {
       png_format = ImageFormat(CL_RGB, CL_UNSIGNED_INT8);
   }
   else if (channel == 4)
   {
       png_format = ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);
   }
   else
   {
       printf("invalid channel, \n");
       exit(1);
   }
   

   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Image2D input_image(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, png_format, srcImgW, srcImgH, 0, srcImgData, &err);
   

   cl::Image2D output_image(context, CL_MEM_WRITE_ONLY, png_format, dstImgW, dstImgH, 0 ,nullptr,&err);
   if (err < 0) {
       printf("Couldn't create image, %s\n", getErrorString(err));
       exit(1);
   }

   {
       ::size_t tmp;
       output_image.getImageInfo(CL_IMAGE_WIDTH, &tmp);
       cout << "image width:" << tmp << endl;

       output_image.getImageInfo(CL_IMAGE_HEIGHT, &tmp);
       cout << "image height:" << tmp << endl;

       output_image.getImageInfo(CL_IMAGE_DEPTH, &tmp);
       cout << "image depth:" << tmp << endl;

       output_image.getImageInfo(CL_IMAGE_ELEMENT_SIZE, &tmp);
       cout << "image ele size:" << tmp << endl;

       output_image.getImageInfo(CL_IMAGE_FORMAT, &tmp);
       cout << "image format:" << tmp << endl; 

       output_image.getImageInfo(CL_IMAGE_ROW_PITCH, &tmp);
       cout << "image pitch:" << tmp << endl;
   }

   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, input_image);
   err |= kernel.setArg(1, output_image);
   if(err < 0) {
       printf("Couldn't set the kernel argument, %s\n", getErrorString(err));
      exit(1);   
   }         
       

   
   cl::NDRange gloabl(dstImgW, dstImgH);
   //err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,  NULL, 0, NULL, NULL);
   err = cq.enqueueNDRangeKernel(kernel, cl::NullRange, gloabl, cl::NullRange);
   if (err < 0) {
       printf("Couldn't exec kernel, %s\n", getErrorString(err));
       exit(1);
   }
   //err = cq.finish();
   
   
   /* Read the result */
   //err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,  result, 0, NULL, NULL);
   cl::size_t<3> origin; origin[0] = 0; origin[1] = 0; origin[2] = 0;
   cl::size_t<3> region; region[0] = dstImgW; region[1] = dstImgH; region[2] = 1;
   err = cq.enqueueReadImage(output_image, CL_TRUE, origin, region, 0,0,dstImgData);
   if(err < 0) {
       printf("Couldn't readimage , %s\n", getErrorString(err));
      exit(1);   
   }

   
   showImage(dstImgData, dstImgW, dstImgH, channel);
   
   
   
   delete[] srcImgData;
   delete[] dstImgData;

   return 0;
}
