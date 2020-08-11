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

const char* kernelSrc = 
    "__kernel void blank(__global float *a) {"
    "}";

#define KERNEL_FUNC "blank"

int main() 
{

   /* OpenCL data structures */   
   cl_int i, j, err;

   /* Data and buffers */
   float full_matrix[80], zero_matrix[80];
   //const size_t buffer_origin[3] = {5*sizeof(float), 3, 0};
   cl::size_t<3>   buffer_origin;
   buffer_origin[0] = 5 * sizeof(float);
   buffer_origin[1] = 3;
   buffer_origin[2] = 0;

   //const size_t host_origin[3] = {1*sizeof(float), 1, 0};
   cl::size_t<3>    host_origin;
   host_origin[0] = 1 * sizeof(float);
   host_origin[1] = 1;
   host_origin[2] = 0;

   //const size_t region[3] = {4*sizeof(float), 4, 1};
   cl::size_t<3>    region;
   region[0] = 4 * sizeof(float);
   region[1] = 4;
   region[2] = 1;


   /* Initialize data */
   for(i=0; i<80; i++) {
      full_matrix[i] = i*1.0f;
      zero_matrix[i] = 0.0;
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
   cl::Program program = cl::Program(context, kernelSrc, false, &errNum);
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

   /* Create a buffer to hold 80 floats */
   //matrix_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(full_matrix), full_matrix, &err);
   cl::Buffer matrix_buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(full_matrix), full_matrix, &err);
   if(err < 0) {
      perror("Couldn't create a buffer object");
      exit(1);   
   }

   /* Set buffer as argument to the kernel */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrix_buffer);
   kernel.setArg(0, matrix_buffer);
   if(err < 0) {
      perror("Couldn't set the buffer as the kernel argument");
      exit(1);   
   }

   
   /* Enqueue kernel */
   //err = clEnqueueTask(queue, kernel, 0, NULL, NULL);
   cq.enqueueTask(kernel, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);   
   }

   /* Enqueue command to write to buffer */
   //err = clEnqueueWriteBuffer(queue, matrix_buffer, CL_TRUE, 0,sizeof(full_matrix), full_matrix, 0, NULL, NULL); 
   err = cq.enqueueWriteBuffer(matrix_buffer, CL_TRUE, 0, sizeof(full_matrix), full_matrix, NULL, NULL);
   if(err < 0) {
      perror("Couldn't write to the buffer object");
      exit(1);   
   }

   /* Enqueue command to read rectangle of data */
   //err = clEnqueueReadBufferRect(queue, matrix_buffer, CL_TRUE,  buffer_origin, host_origin, region, 10*sizeof(float), 0, 
   //      10*sizeof(float), 0, zero_matrix, 0, NULL, NULL);
   err = cq.enqueueReadBufferRect(matrix_buffer, CL_TRUE, buffer_origin, host_origin, region, 10 * sizeof(float), 0,
       10 * sizeof(float), 0, zero_matrix, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the rectangle from the buffer object");
      exit(1);   
   }

   /* Display updated buffer */
   for(i=0; i<8; i++) {
      for(j=0; j<10; j++) {
         printf("%6.1f", zero_matrix[j+i*10]);
      }
      printf("\n");
   }
   return 0;
}