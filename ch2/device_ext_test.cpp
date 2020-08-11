#include <stdio.h>
#include <stdlib.h>
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

int main() {

    cl::Platform platform = cl::Platform::getDefault();

    vector<cl::Device>  devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    int deviceIndex = 0;
    /* Obtain data for each connected device */
    for(auto device: devices) 
    {
        string infoValue;

        device.getInfo(CL_DEVICE_NAME, &infoValue);
        cout << "device  " << deviceIndex << " name " << infoValue << endl;

        device.getInfo(CL_DEVICE_VENDOR, &infoValue);
        cout << "device  " << deviceIndex << " vendor " << infoValue << endl;

        device.getInfo(CL_DEVICE_EXTENSIONS, &infoValue);
        cout << "device  " << deviceIndex << " ext " << infoValue << endl;


        cl_ulong    memSize;
        device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &memSize);
        cout << "device  " << deviceIndex << " mem " << memSize/(1024*0124) << "M" << endl;

        cl_uint bits;
        device.getInfo(CL_DEVICE_ADDRESS_BITS, &bits);
        cout << "device  " << deviceIndex << " bits " << bits << endl;


        deviceIndex++;
    }

    
    return 0;
}