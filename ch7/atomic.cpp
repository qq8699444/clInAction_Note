#define _CRT_SECURE_NO_WARNINGS


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <stddef.h>
#include <thread>
#include <mutex>
#include <condition_variable>

#ifdef MAC
#include <OpenCL/cl.h>
#else  
#include <CL/cl.hpp> 
#endif

using namespace std;
using namespace cl;

using std::size_t;

const std::string kernelCode = R"(
__kernel void atomic( __global int* buf)
{
	__local int a,b ;
	
	a = 0;
	b = 0;
	
	
	a++;
	
	atomic_inc(&b);
	
	
	buf[2*get_global_id(0)] = a;
	buf[2*get_global_id(0)+1] = b;
}

)";

#define KERNEL_FUNC "atomic"



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
    int data[8];
    cl_int err;
    cl::Event   prof_event;
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer res_buff(context, CL_MEM_WRITE_ONLY, sizeof(data), nullptr, &err);

   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, res_buff);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }   


 
   NDRange    globalSize(4);
   NDRange    localSize(4);
   err = cq.enqueueNDRangeKernel(kernel, NullRange, globalSize, localSize);
   if (err < 0) {
       perror("Couldn't enqueue the kernel execution command");
       exit(1);
   }

   cq.finish();

   cq.enqueueReadBuffer(res_buff, CL_TRUE, 0, sizeof(data), data);
  
   for (int i = 0;i < 4;i++)
   {
       cout << data[2 * i] << ", " << data[2 * i + 1] << endl;
   }

   return 0;
}
