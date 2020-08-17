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
__kernel void callback(__global float *buffer) {
   float4 five_vector = (float4)(5.0);

   for(int i=0; i<1024; i++) {
      vstore4(five_vector, i, buffer);
   }
}
 
)";

#define KERNEL_FUNC "callback"

std::mutex mtx;
std::condition_variable kernel_cv;
std::condition_variable read_cv;
void CL_CALLBACK kernel_complete(cl_event e, cl_int status, void* data) {
    printf("%s\n", (char*)data);

    std::unique_lock <std::mutex> lck(mtx);
    kernel_cv.notify_all();
}

void CL_CALLBACK read_complete(cl_event e, cl_int status, void* data) {
    cout << "read callback .." << endl;
    std::unique_lock <std::mutex> lck(mtx);
    read_cv.notify_all();
}

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
    float data[4096];
   cl_int err;
   cl::Event   kernel_event;
   cl::Event   read_event;
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer res_buff(context, CL_MEM_WRITE_ONLY, sizeof(data), NULL, NULL);

   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, res_buff);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }   

   err = cq.enqueueTask(kernel,NULL,&kernel_event);
   if(err < 0) {
      perror("Couldn't enqueue the kernel execution command");
      exit(1);   
   }

   const  char *kernel_msg = "The kernel finished successfully.\n\0";
   err = kernel_event.setCallback(CL_COMPLETE, kernel_complete, (void*)kernel_msg);
   if (err < 0) {
       perror("Couldn't set callback");
       exit(1);
   }

   {
       std::unique_lock <std::mutex> lck(mtx);
       kernel_cv.wait(lck);
       cout << "wait kernel .." << endl;
   }

   /* Read the result */
   //err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,  result, 0, NULL, NULL);
   err = cq.enqueueReadBuffer(res_buff, CL_FALSE, 0, sizeof(data) , data, nullptr, &read_event);
   if(err < 0) {
      perror("Couldn't enqueue the read buffer command");
      exit(1);   
   }

   read_event.setCallback(CL_COMPLETE, read_complete, data);
   
   
   
   {
       std::unique_lock <std::mutex> lck(mtx);
       read_cv.wait(lck);
       cout << "wait read .." << endl;
   }

   bool check = true;
   for (int i = 0; i < 4096; i++) {
       if (data[i] != 5.0) {
           check = CL_FALSE;
           break;
       }
   }
   if (check)
       printf("The data has been initialized successfully.\n");
   else
       printf("The data has not been initialized successfully.\n");
   return 0;
}
