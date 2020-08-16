#define _CRT_SECURE_NO_WARNINGS


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <stddef.h>



#include <CL/cl.hpp> 


using namespace std;
using namespace cl;

using std::size_t;
#define PI       3.14159265358979323846 
const std::string polarRectCode = R"(
__kernel void polar_rect(__global float4 *r_vals, 
                         __global float4 *angles,
                         __global float4 *x_coords, 
                         __global float4 *y_coords) {

   *y_coords = sincos(*angles, x_coords);
   *x_coords *= *r_vals;
   *y_coords *= *r_vals;
}
)";

#define KERNEL_FUNC "polar_rect"

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
    cl::Program program = cl::Program(context, polarRectCode, false, &errNum);
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
    float r_coords[4] = { 2, 1, 3, 4 };
    float angles[4] = { 3 * PI / 8, 3 * PI / 4, 4 * PI / 3, 11 * PI / 6 };
    float x_coords[4], y_coords[4];
    cl_int err;
   
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer r_coords_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(r_coords), r_coords, NULL);
   cl::Buffer angles_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(angles), angles, NULL);

   cl::Buffer x_coords_buffer(context, CL_MEM_WRITE_ONLY,sizeof(x_coords), NULL, NULL);   
   cl::Buffer y_coords_buffer(context, CL_MEM_WRITE_ONLY, sizeof(y_coords), NULL, NULL);
   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, r_coords_buffer);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }         
       
   kernel.setArg(1, angles_buffer);
   kernel.setArg(2, x_coords_buffer);
   kernel.setArg(3, y_coords_buffer);
   
   err = cq.enqueueTask(kernel);
   if(err < 0) {
      perror("Couldn't enqueue the kernel execution command");
      exit(1);   
   }

   /* Read the result */
   //err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,  result, 0, NULL, NULL);
   err  = cq.enqueueReadBuffer(x_coords_buffer, CL_TRUE, 0, sizeof(x_coords), x_coords);
   err |= cq.enqueueReadBuffer(y_coords_buffer, CL_TRUE, 0, sizeof(y_coords), y_coords);
   if(err < 0) {
      perror("Couldn't enqueue the read buffer command");
      exit(1);   
   }

   /* Display the results */
   for (int i = 0; i < 4; i++) {
       printf("(%6.3f, %6.3f)\n", x_coords[i], y_coords[i]);
   }

   return 0;
}
