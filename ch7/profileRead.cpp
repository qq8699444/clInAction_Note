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
__kernel void profile_read(__global char16 *c, int num) {

   for(int i=0; i<num; i++) {
      c[i] = (char16)(5);
   }
}

)";

#define KERNEL_FUNC "profile_read"

#define NUM_BYTES 131072
#define NUM_ITERATIONS 2000
#define PROFILE_READ 1

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
    float data[NUM_BYTES];
   cl_int err;   
   cl::Event   prof_event;
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer res_buff(context, CL_MEM_WRITE_ONLY, sizeof(data), NULL, NULL);

   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, res_buff);
   cl_int num_vector = NUM_BYTES / 16;
   err |= kernel.setArg(1, num_vector);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }   


   cl_ulong total_time = 0;
   for (int i = 0; i < NUM_ITERATIONS; i++)
   {
       err = cq.enqueueTask(kernel);
       if (err < 0) {
           perror("Couldn't enqueue the kernel execution command");
           exit(1);
       }


   
    
#if PROFILE_READ
       /* Read the result */
       //err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,  result, 0, NULL, NULL);
       err = cq.enqueueReadBuffer(res_buff, CL_TRUE, 0, sizeof(data) , data, nullptr, &prof_event);
       if(err < 0) {
          perror("Couldn't enqueue the read buffer command");
          exit(1);   
       }
  
#else
       void* mapped_memory;
       mapped_memory = cq.enqueueMapBuffer(res_buff, CL_TRUE, CL_MAP_READ, 0, sizeof(data),NULL, &prof_event,&err);
       if (err < 0) {
           perror("Couldn't enqueue the map buffer command");
           exit(1);
       }
#endif
       cl_ulong time_start, time_end;
       prof_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
       prof_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
       total_time += time_end - time_start;

#if PROFILE_READ

#else
       err = cq.enqueueUnmapMemObject(res_buff, mapped_memory);
       if (err < 0) {
           perror("Couldn't unmap the buffer");
           exit(1);
       }
#endif
   }
  
#if  PROFILE_READ
   cout << "Average read time: "  << total_time / NUM_ITERATIONS << endl;
#else
   cout << "Average map time: " << total_time / NUM_ITERATIONS << endl;
#endif

   return 0;
}
