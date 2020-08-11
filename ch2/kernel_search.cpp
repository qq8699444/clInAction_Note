#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "test.cl"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <iostream>
#include <string>
#include <vector>


#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif

using namespace std;


const std::string Kernelscode = R"(
__kernel void add(__global float *a,
                  __global float *b,
                  __global float *c) {
   
   *c = *a + *b;
}

__kernel void sub(__global float *a,
                  __global float *b,
                  __global float *c) {
   
   *c = *a - *b;
}

__kernel void mult(__global float *a,
                   __global float *b,
                   __global float *c) {
   
   *c = *a * *b;
}

__kernel void div(__global float *a,
                  __global float *b,
                  __global float *c) {
   
   *c = *a / *b;
}
)";

int main() {

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
    cl::Program program = cl::Program(context, Kernelscode, false, &errNum);
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


    vector<cl::Kernel>   kernels;
    program.createKernels(&kernels);

    size_t num_kernels = kernels.size();
   /* Search for the named kernel */
   for(size_t i=0; i<num_kernels; i++) {	
       cl::Kernel& kernel = kernels[i];
       string kernelName;
       kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernelName);

       cl_uint argCnt;
       kernel.getInfo(CL_KERNEL_NUM_ARGS, &argCnt);
       cout << "find kernel:" << kernelName  << ",num args:" << argCnt  << endl;
   }									

   return 0;
}