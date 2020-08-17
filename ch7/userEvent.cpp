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
__kernel void user_event(__global float4 *v) {

   *v *= -1.0f;
}
)";

#define KERNEL_FUNC "user_event"

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
   float data[4];
   float result[4];
   for (int i = 0; i < 4; i++)
       data[i] = i * 1.0f + 1.0f;

   cl_int err;
   cl::Event   kernel_event;
   cl::Event   read_event;
   cl::UserEvent    user_event(context,&err);
   if (err < 0) {
       perror("Couldn't create user event");
       exit(1);
   }
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer res_buff(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data), data, NULL);

   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, res_buff);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }   

   std::vector<Event> events;
   events.push_back(user_event);
   err = cq.enqueueTask(kernel,&events,&kernel_event);
   if(err < 0) {
      perror("Couldn't enqueue the kernel execution command");
      exit(1);
   }   

   /* Read the result */
   //err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,  result, 0, NULL, NULL);
   std::vector<Event> events2;
   events.push_back(kernel_event);
   err = cq.enqueueReadBuffer(res_buff, CL_FALSE, 0, sizeof(result) , result, &events2, &read_event);
   if(err < 0) {
      perror("Couldn't enqueue the read buffer command");
      exit(1);   
   }

   read_event.setCallback(CL_COMPLETE, read_complete, nullptr);
   
   printf("Press ENTER to continue.\n");
   getchar();

   user_event.setStatus(CL_COMPLETE);
   
   {
       std::unique_lock <std::mutex> lck(mtx);
       read_cv.wait(lck);
       cout << "wait read .." << endl;
   }

   printf("%f %f %f %f\n", result[0], result[1], result[2], result[3]);
   return 0;
}
