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

const char* kernelSrc = R"(
__kernel void hello_kernel(__global char16 *msg) {
   *msg = (char16)('H', 'e', 'l', 'l', 'o', ' ',
      'k', 'e', 'r', 'n', 'e', 'l', '!', '!', '!', '\0');
}
)";

#define KERNEL_FUNC "hello_kernel"

int main()
{
    cl_int err;

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

    char msg[16];
    cl::Buffer msg_buffer(context, CL_MEM_WRITE_ONLY, sizeof(msg), NULL, &err);
    if (err < 0) {
        perror("Couldn't create a buffer");
        exit(1);
    }

    kernel.setArg(0, msg_buffer);
    cq.enqueueTask(kernel);

    cq.enqueueReadBuffer(msg_buffer, true, 0, sizeof(msg), msg);
    printf("Kernel output: %s\n", msg);
    return 0;
}