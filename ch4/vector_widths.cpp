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

    for (auto device : devices)
    {
        cl_uint vector_width;

        device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, &vector_width);
        cout << "referred vector width in chars:" << vector_width << endl;

        device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, &vector_width);
        cout << "Preferred vector width in shorts: " << vector_width << endl;

        device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, &vector_width);
        cout << "Preferred vector width in ints: " << vector_width << endl;

        device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, &vector_width);
        cout << "Preferred vector width in longs: " << vector_width << endl;

        device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,&vector_width);
        cout << "Preferred vector width in floats: " <<  vector_width << endl;

        device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,&vector_width);
        cout << "Preferred vector width in doubles:" <<  vector_width << endl;
    }
   

   
   return 0;
}