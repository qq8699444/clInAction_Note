#define _CRT_SECURE_NO_WARNINGS
#define NUM_FILES 2
#define PROGRAM_FILE_1 "good.cl"
#define PROGRAM_FILE_2 "bad.cl"

#include <stdio.h>
#include <stdlib.h>
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


const std::string goodKernel = R"(
__kernel void good(__global float *a,
                   __global float *b,
                   __global float *c) {
   
   *c = *a + *b;
}
)";


const std::string badKernel = R"(
__kernel void bad(__global float *a,
                   __global float *b,
                   __global float *c) {
   
   *c = *a + *b;1
}

)";


int main() {

    cl::Platform platform = cl::Platform::getDefault();
    
    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, NULL);

    cl_int errNum = CL_SUCCESS;
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    /* Create a CL command queue for the device*/
    cl::CommandQueue cq = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &errNum);
    if (errNum != CL_SUCCESS) {
        printf("failed to get queue£º%d\n", errNum);
        return -1;
    }

    string kernels[2] = { goodKernel,badKernel };
    for (auto kcode:kernels)
    {
        cout << "kcode  " << kcode << endl;
        /* Program/kernel data structures */
        cl::Program program = cl::Program(context, kcode, false, &errNum);
        errNum = program.build(devices);  //±àÒëProgram
        if (errNum < 0)
        {
            std::size_t log_size;
            program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log_size);
            /* Find size of log and print to std output */

            string buildlog;
            program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &buildlog);   
            cout << "buildlog : " << buildlog << endl;
        }
    }
    printf("-----\n");
    return 0;
}