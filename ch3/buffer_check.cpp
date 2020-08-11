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


int main() {

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

    float main_data[256];
    cl_int  err;
   /* Create a buffer to hold 100 floating-point values */
   //main_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(main_data), main_data, &err);
   cl::Buffer main_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(main_data), main_data, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   }

   /* Create a sub-buffer containing values 30-49 */
   cl_buffer_region region;
   region.origin = 30*sizeof(float);
   region.size = 20*sizeof(float);
   ///sub_buffer = clCreateSubBuffer(main_buffer, CL_MEM_READ_ONLY , CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
   cl::Buffer  sub_buffer = main_buffer.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
   if(err < 0) {
      perror("Couldn't create a sub-buffer");
      exit(1);   
   }

   /* Obtain size information about the buffers */
   size_t main_buffer_size, sub_buffer_size;
   //clGetMemObjectInfo(main_buffer, CL_MEM_SIZE, sizeof(main_buffer_size), &main_buffer_size, NULL);
   main_buffer.getInfo(CL_MEM_SIZE, &main_buffer_size);
   //clGetMemObjectInfo(sub_buffer, CL_MEM_SIZE,  sizeof(sub_buffer_size), &sub_buffer_size, NULL);
   sub_buffer.getInfo(CL_MEM_SIZE, &sub_buffer_size);
   cout << "Main buffer size: " <<  main_buffer_size << endl;
   cout << "Sub-buffer size:  " << sub_buffer_size << endl;
   
   /* Obtain the host pointers */
   void *main_buffer_mem = NULL, *sub_buffer_mem = NULL;
   //clGetMemObjectInfo(main_buffer, CL_MEM_HOST_PTR, sizeof(main_buffer_mem),  &main_buffer_mem, NULL);
   //clGetMemObjectInfo(sub_buffer, CL_MEM_HOST_PTR, sizeof(sub_buffer_mem),    &sub_buffer_mem, NULL);
   main_buffer.getInfo(CL_MEM_HOST_PTR, &main_buffer_mem);
   sub_buffer.getInfo(CL_MEM_HOST_PTR, &sub_buffer_mem);
   printf("Main buffer memory address: %p\n", main_buffer_mem);
   printf("Sub-buffer memory address:  %p\n", sub_buffer_mem);

   /* Print the address of the main data */
   printf("Main array address: %p\n", main_data);
   return 0;
}