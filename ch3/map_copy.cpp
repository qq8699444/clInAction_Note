#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <iostream>
#include <string>
#include <vector>


#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif

using namespace std;

const char* blankKernel =
"__kernel void blank(__global float *a, __global float *b) {"
"}";
#define KERNEL_FUNC "blank"

int main() {

   /* OpenCL data structures */
   
   cl_int i, j, err;

   /* Data and buffers */
   float data_one[100], data_two[100], result_array[100];   
   void* mapped_memory;

   /* Initialize arrays */
   for(i=0; i<100; i++) {
      data_one[i] = 1.0f*i;
      data_two[i] = -1.0f*i;
      result_array[i] = 0.0f;
   }

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
   cl::Program program = cl::Program(context, blankKernel, false, &errNum);
   errNum = program.build(devices);  //±‡“ÎProgram
   if (errNum < 0)
   {
       std::size_t log_size;
       program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log_size);
       /* Find size of log and print to std output */

       char* program_log = new char[log_size + 1];
       program_log[log_size] = '\0';

       program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, program_log);
       printf("%s\n", program_log);
       delete[] program_log;
       return -2;
   }

   cl::Kernel kernel(program, KERNEL_FUNC);

   /* Create buffers */
   //buffer_one = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data_one), data_one, &err);
   cl::Buffer buffer_one(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data_one), data_one, &err);
   if(err < 0) {
      perror("Couldn't create a buffer object");
      exit(1);   
   }
   
   //buffer_two = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data_two), data_two, NULL);
   cl::Buffer buffer_two(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data_two), data_two, &err);

   /* Set buffers as arguments to the kernel */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_one);
   err = kernel.setArg(0, buffer_one);
   //err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_two);
   err |= kernel.setArg(1, buffer_two);
   if(err < 0) {
      perror("Couldn't set the buffer as the kernel argument");
      exit(1);   
   }

   /* Enqueue kernel */
   //err = clEnqueueTask(queue, kernel, 0, NULL, NULL);
   err = cq.enqueueTask(kernel);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);   
   }

   /* Enqueue command to copy buffer one to buffer two */
   //err = clEnqueueCopyBuffer(queue, buffer_one, buffer_two, 0, 0,sizeof(data_one), 0, NULL, NULL); 
   err = cq.enqueueCopyBuffer(buffer_one, buffer_two, 0, 0, sizeof(data_one));
   if(err < 0) {
      perror("Couldn't perform the buffer copy");
      exit(1);   
   }

   /* Enqueue command to map buffer two to host memory */
   //mapped_memory = clEnqueueMapBuffer(queue, buffer_two, CL_TRUE, CL_MAP_READ, 0, sizeof(data_two), 0, NULL, NULL, &err);
   mapped_memory = cq.enqueueMapBuffer(buffer_two, CL_TRUE, CL_MAP_READ, 0, sizeof(data_two), NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't map the buffer to host memory");
      exit(1);   
   }

   /* Transfer memory and unmap the buffer */
   memcpy(result_array, mapped_memory, sizeof(data_two));
   //err = clEnqueueUnmapMemObject(queue, buffer_two, mapped_memory,         0, NULL, NULL);
   err =cq.enqueueUnmapMemObject(buffer_two, mapped_memory, NULL, NULL);
   if(err < 0) {
      perror("Couldn't unmap the buffer");
      exit(1);   
   }

   /* Display updated buffer */
   for(i=0; i<10; i++) {
      for(j=0; j<10; j++) {
         printf("%6.1f", result_array[j+i*10]);
      }
      printf("\n");
   }

   return 0;
}