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

#include "geterror.h"
using namespace std;
using namespace cl;

using std::size_t;

const std::string kernelCode = R"(
__kernel void idCheck(__global float* output) {
   
    size_t global_id_0 = get_global_id(0);
    size_t global_id_1 = get_global_id(1);
    
    size_t global_size_0 = get_global_size(0);
    
    size_t offset_0 = get_global_offset(0);
	size_t offset_1 = get_global_offset(1);
	
	size_t local_id_0 = get_local_id(0);
	size_t local_id_1 = get_local_id(1);
	
	int index_0 = global_id_0 - offset_0;
	int index_1 = global_id_1 - offset_1;
	int index = index_1 * global_size_0 + index_0;
	
	float f = global_id_0*10.0f + global_id_1*1.0f + local_id_0*0.1f  + local_id_1*0.01f;
   	output[index] = f;		
}
)";

const string kernelFunc = "idCheck";

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

    cl::Kernel kernel(program, kernelFunc.c_str());
   
   /* Data and buffers */
   float  result[6*4];   
   cl_int err;   
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer res_buff(context, CL_MEM_WRITE_ONLY, sizeof(result), NULL, NULL);

   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, res_buff);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }         
       

   auto dim = 2;

   cl::NDRange  global_offset(3, 5);   
   cl::NDRange  global_size(6, 4);
   cl::NDRange  local_size(3, 2 );
   err = cq.enqueueNDRangeKernel(kernel, global_offset, global_size, local_size);
   if(err < 0) {
      printf("Couldn't enqueue the kernel execution command ,%s", getErrorString(err));
      exit(1);   
   }

   /* Read the result */
   //err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,  result, 0, NULL, NULL);
   err = cq.enqueueReadBuffer(res_buff, CL_TRUE, 0, sizeof(result), result);
   if(err < 0) {
      perror("Couldn't enqueue the read buffer command");
      exit(1);   
   }

   /* Test the result */
   for (int r = 0; r < 4;r++)    
   {
       printf("%2.2f %2.2f %2.2f %2.2f %2.2f %2.2f\n", 
           result[r * 6 + 0], result[r * 6 + 1], result[r * 6 + 2],
           result[r * 6 + 3], result[r * 6 + 4], result[r * 6 + 5]);
   }

   return 0;
}
