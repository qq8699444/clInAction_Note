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


const std::string vectorBytesKernel = R"(
__kernel void vector_bytes(__global uchar16 *test) {

   /* Initialize a vector of four integers */
   uint4 vec = {0x00010203, 0x04050607, 
      0x08090A0B, 0x0C0D0E0F}; 

   /* Convert the uint4 to a uchar16 byte-by-byte */
   uchar *p = &vec;
   *test = (uchar16)(*p, *(p+1), *(p+2), *(p+3), *(p+4), *(p+5), 
      *(p+6), *(p+7), *(p+8), *(p+9), *(p+10), *(p+11), *(p+12), 
      *(p+13), *(p+14), *(p+15));
}

)";

#define KERNEL_FUNC "vector_bytes"

int main()
{
    cl_int err;
    float a = 6.0, b = 2.0, result = 3;

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
    
    unsigned char test[16];
    cl::Buffer buffer_o(context, CL_MEM_WRITE_ONLY, sizeof(test), nullptr, &err);

    cl::Program program = cl::Program(context, vectorBytesKernel, false, &errNum);
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
    kernel.setArg(0, buffer_o);

    cq.enqueueTask(kernel);

    cq.enqueueReadBuffer(buffer_o, true, 0, sizeof(test), test);
    for (int i = 0; i < 16; i++) {
        printf("0x%X, ", test[i]);
    }
    

    return 0;
}