#define _CRT_SECURE_NO_WARNINGS


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <stddef.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else  
#include <CL/cl.hpp> 
#endif

using namespace std;
using namespace cl;

using std::size_t;

const std::string modRoundCode = R"(
__kernel void mod_round(__global float *mod_input, 
                        __global float *mod_output, 
                        __global float4 *round_input,
                        __global float4 *round_output) {

   /* Use fmod and remainder: 317.0, 23.0 */
   mod_output[0] = fmod(mod_input[0], mod_input[1]);
   mod_output[1] = remainder(mod_input[0], mod_input[1]);
   
   /* Rounds the input values: -6.5, -3.5, 3.5, and 6.5 */
   round_output[0] = rint(*round_input);      
   round_output[1] = round(*round_input);
   round_output[2] = ceil(*round_input);
   round_output[3] = floor(*round_input);
   round_output[4] = trunc(*round_input);   
}

)";

#define KERNEL_FUNC "mod_round"

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
    cl::Program program = cl::Program(context, modRoundCode, false, &errNum);
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
    float mod_input[2] = { 317.0f, 23.0f };
    float mod_output[2];
    float round_input[4] = { -6.5f, -3.5f, 3.5f, 6.5f };
    float round_output[20];
   cl_int err;
   
   
   //res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, NULL);
   cl::Buffer mod_input_buff(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mod_input), mod_input, NULL);
   cl::Buffer mod_output_buffer(context, CL_MEM_WRITE_ONLY,sizeof(mod_output), NULL, NULL);

   cl::Buffer round_input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(round_input), round_input, NULL);
   cl::Buffer round_output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(round_output), NULL, NULL);
   /* Create kernel arguments from the CL buffers */
   //err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
   err = kernel.setArg(0, mod_input_buff);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);   
   }         
       
   kernel.setArg(1, mod_output_buffer);
   kernel.setArg(2, round_input_buffer);
   kernel.setArg(3, round_output_buffer);
   
   err = cq.enqueueTask(kernel);
   if(err < 0) {
      perror("Couldn't enqueue the kernel execution command");
      exit(1);   
   }

   /* Read the result */
   //err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,  result, 0, NULL, NULL);
   err  = cq.enqueueReadBuffer(mod_output_buffer, CL_TRUE, 0, sizeof(mod_output), mod_output);
   err |= cq.enqueueReadBuffer(round_output_buffer, CL_TRUE, 0, sizeof(round_output), round_output);
   if(err < 0) {
      perror("Couldn't enqueue the read buffer command");
      exit(1);   
   }

   /* Display data */
   printf("fmod(%.1f, %.1f)      = %.1f\n", mod_input[0], mod_input[1], mod_output[0]);
   printf("remainder(%.1f, %.1f) = %.1f\n\n", mod_input[0], mod_input[1], mod_output[1]);

   printf("Rounding input: %.1f %.1f %.1f %.1f\n",
       round_input[0], round_input[1], round_input[2], round_input[3]);
   printf("rint:  %.1f, %.1f, %.1f, %.1f\n",
       round_output[0], round_output[1], round_output[2], round_output[3]);
   printf("round: %.1f, %.1f, %.1f, %.1f\n",
       round_output[4], round_output[5], round_output[6], round_output[7]);
   printf("ceil:  %.1f, %.1f, %.1f, %.1f\n",
       round_output[8], round_output[9], round_output[10], round_output[11]);
   printf("floor: %.1f, %.1f, %.1f, %.1f\n",
       round_output[12], round_output[13], round_output[14], round_output[15]);
   printf("trunc: %.1f, %.1f, %.1f, %.1f\n",
       round_output[16], round_output[17], round_output[18], round_output[19]);


   return 0;
}
