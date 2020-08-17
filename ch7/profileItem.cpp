#define _CRT_SECURE_NO_WARNINGS


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <stddef.h>
#include <thread>
#include <mutex>
#include <condition_variable>

#ifdef MAC
#include <OpenCL/cl.h>
#else  
#include <CL/cl.hpp> 
#endif

using namespace std;
using namespace cl;

using std::size_t;

const std::string kernelCode = R"(
__kernel void profile_items(__global int4 *x, int num_ints) {

   int num_vectors = num_ints/(4 * get_global_size(0));

   x += get_global_id(0) * num_vectors;
   for(int i=0; i<num_vectors; i++) {
      x[i] += 1;
      x[i] *= 2;
      x[i] /= 3;
   }
}
)";

#define KERNEL_FUNC "profile_items"

#define NUM_INTS 4096
#define NUM_ITEMS 64
#define NUM_ITERATIONS 2000

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
    cl::Program program = cl::Program(context, kernelCode, false, &errNum);
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
    int data[NUM_INTS];
    cl_int err;
    cl::Event   prof_event;
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer res_buff(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data), data, &err);

   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, res_buff);
   err |= kernel.setArg(1, NUM_INTS);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }   


   cl_ulong total_time = 0;
   for (int i = 0; i < NUM_ITERATIONS; i++)
   {
       NDRange    globalSize(NUM_ITEMS);
       err = cq.enqueueNDRangeKernel(kernel, NullRange, globalSize, NullRange, nullptr, &prof_event);
       if (err < 0) {
           perror("Couldn't enqueue the kernel execution command");
           exit(1);
       }

       cq.finish();

       cl_ulong time_start, time_end;
       prof_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
       prof_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
       total_time += time_end - time_start;
   }
  

   cout << "Average map time: " << total_time / NUM_ITERATIONS << endl; 


   return 0;
}
