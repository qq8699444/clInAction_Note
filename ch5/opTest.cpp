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

using namespace std;
using namespace cl;

using std::size_t;

const std::string kernelCode = R"(
__kernel void opTest(__global int4* output) {
   
   int4 vec = (int4)(1,2,3,4);
   vec += 4;
 
   if (vec.s2 == 7)
   	vec &= (int4)(-1,-1,0,-1);
   	
   vec.s01 = vec.s23 < 7;
   
   while(vec.s3 > 7 && (vec.s0 < 16) && (vec.s1 < 16))
   	vec.s3 >>= 1;
   	
   	*output = vec;	
}
)";

#define KERNEL_FUNC "opTest"

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

    cl::Kernel kernel(program, KERNEL_FUNC);
    

  
  
   
   /* Data and buffers */
   int  result[4];   
   cl_int err;
   
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer res_buff(context, CL_MEM_WRITE_ONLY, sizeof(int) * 4, NULL, NULL);

   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, res_buff);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }         
       

   
   err = cq.enqueueTask(kernel);
   if(err < 0) {
      perror("Couldn't enqueue the kernel execution command");
      exit(1);   
   }

   /* Read the result */
   //err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,  result, 0, NULL, NULL);
   err = cq.enqueueReadBuffer(res_buff, CL_TRUE, 0, sizeof(int) * 4, result);
   if(err < 0) {
      perror("Couldn't enqueue the read buffer command");
      exit(1);   
   }

   /* Test the result */
   for (int i = 0; i < 4;i++)
   {
       cout << "result[" << i << "]:" << result[i] << endl;
   }

   return 0;
}
