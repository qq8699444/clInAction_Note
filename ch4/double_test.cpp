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


const std::string doubleTestKernel = R"(
/* Enable the double type if available */
#ifdef FP_64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void double_test(__global float* a,
                          __global float* b,
                          __global float* out) {

/* Declare a double variable if possible */
#ifdef FP_64
   double c = (double)(*a / *b);
   *out = (float)c;
   
/* Use floats if doubles are unavailable */
#else
   *out = *a * *b;
#endif
}

)";

#define KERNEL_FUNC "double_test"

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
        printf("failed to get queue£º%d\n", errNum);
        return -1;
    }

    cl::Buffer buffer_a(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), &a, &err);
    if (err < 0) {
        perror("Couldn't create a buffer object");
        exit(1);
    }

    //buffer_two = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data_two), data_two, NULL);
    cl::Buffer buffer_b(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), &b, &err);

    cl::Buffer buffer_o(context, CL_MEM_WRITE_ONLY , sizeof(float), nullptr, &err);


    string buildOpts[2] = { "", "-DFP_64" };
    for (auto& opt:buildOpts)
    {
        /* Program/kernel data structures */
        cl::Program program = cl::Program(context, doubleTestKernel, false, &errNum);
        errNum = program.build(devices,opt.c_str());  //±àÒëProgram
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
        kernel.setArg(0, buffer_a);
        kernel.setArg(1, buffer_b);
        kernel.setArg(2, buffer_o);
       
        cq.enqueueTask(kernel);

        cq.enqueueReadBuffer(buffer_o, true, 0, sizeof(float), &result);
        cout << "result:" << result << endl;
    }

    return 0;
}