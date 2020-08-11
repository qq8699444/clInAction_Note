#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>


#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif

using namespace std;

int main() {

    cl::Platform platforms = cl::Platform::getDefault();

    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, NULL);

    cl_int errNum = CL_SUCCESS;
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    for (auto device:devices)
    {
        cl_device_fp_config flag;
        device.getInfo(CL_DEVICE_SINGLE_FP_CONFIG, &flag);

        printf("Float Processing Features:\n");
        if (flag & CL_FP_INF_NAN)
            printf("INF and NaN values supported.\n");
        if (flag & CL_FP_DENORM)
            printf("Denormalized numbers supported.\n");
        if (flag & CL_FP_ROUND_TO_NEAREST)
            printf("Round To Nearest Even mode supported.\n");
        if (flag & CL_FP_ROUND_TO_INF)
            printf("Round To Infinity mode supported.\n");
        if (flag & CL_FP_ROUND_TO_ZERO)
            printf("Round To Zero mode supported.\n");
        if (flag & CL_FP_FMA)
            printf("Floating-point multiply-and-add operation supported.\n");

#ifndef MAC
        if (flag & CL_FP_SOFT_FLOAT)
            printf("Basic floating-point processing performed in software.\n");
#endif
    }
   

   return 0;
}