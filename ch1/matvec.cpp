#define _CRT_SECURE_NO_WARNINGS

#define PROGRAM_FILE "matvec.cl"
#define KERNEL_FUNC "matvec_mult"

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

using namespace std;
using namespace cl;

using std::size_t;

const std::string addKernel = R"(
__kernel void matvec_mult(__global float4* matrix,
                          __global float4* vector,
                          __global float* result) {
   
   int i = get_global_id(0);
   result[i] = dot(matrix[i], vector[0]);
}
)";



int main() 
{
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
    cl::Program program = cl::Program(context, addKernel, false, &errNum);
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

    cl::Kernel matvec_mult_kernel(program, KERNEL_FUNC);
    

  
  
   
   /* Data and buffers */
   float mat[16], vec[4], result[4];
   float correct[4] = {0.0f, 0.0f, 0.0f, 0.0f};

   /* Initialize data to be processed by the kernel */
   for(int i=0; i<16; i++) {
      mat[i] = i * 2.0f;
   } 
   for(int i=0; i<4; i++) {
      vec[i] = i * 3.0f;
      correct[0] += mat[i]    * vec[i];
      correct[1] += mat[i+4]  * vec[i];
      correct[2] += mat[i+8]  * vec[i];
      correct[3] += mat[i+12] * vec[i];      
   } 

   
   cl_int err;
   /* Create CL buffers to hold input and output data */
   //mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*16, mat, &err);
   cl::Buffer mat_buff(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * 16, mat, &err);
   if(err < 0) {
      perror("Couldn't create a buffer object");
      exit(1);   
   }  

   //vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*4, vec, NULL);
   cl::Buffer vec_buff(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * 4, vec, NULL);
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer res_buff(context, CL_MEM_WRITE_ONLY, sizeof(float) * 4, NULL, NULL);

   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = matvec_mult_kernel.setArg(0, mat_buff);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }         
   //clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buff);
   matvec_mult_kernel.setArg(1, vec_buff);
   //clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff);
   matvec_mult_kernel.setArg(2, res_buff);
     

   /* Enqueue the command queue to the device */
   auto work_units_per_kernel = 4; /* 4 work-units per kernel */ 
   cl::NDRange gloabl(work_units_per_kernel);
   //err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,  NULL, 0, NULL, NULL);
   err = cq.enqueueNDRangeKernel(matvec_mult_kernel, cl::NullRange, gloabl, cl::NullRange);
   if(err < 0) {
      perror("Couldn't enqueue the kernel execution command");
      exit(1);   
   }

   /* Read the result */
   //err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,  result, 0, NULL, NULL);
   err = cq.enqueueReadBuffer(res_buff, CL_TRUE, 0, sizeof(float) * 4, result);
   if(err < 0) {
      perror("Couldn't enqueue the read buffer command");
      exit(1);   
   }

   /* Test the result */
   if((result[0] == correct[0]) && (result[1] == correct[1])
      && (result[2] == correct[2]) && (result[3] == correct[3])) {
      printf("Matrix-vector multiplication successful.\n");
   }
   else {
      printf("Matrix-vector multiplication unsuccessful.\n");
   }  

   return 0;
}
