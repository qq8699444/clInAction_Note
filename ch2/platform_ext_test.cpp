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
int main() 
{
    const string icd_ext = "cl_khr_icd";
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    int platform_index = 0;
   /* Find extensions of all platforms */
   for(auto platform : platforms) 
   {
       string extInfo;
       cl_int err;

       string infoValue;

       platform.getInfo(CL_PLATFORM_NAME, &infoValue);
       cout << "Platform  " << platform_index << " name " << infoValue << endl;

       platform.getInfo(CL_PLATFORM_VENDOR, &infoValue);
       cout << "Platform  " << platform_index << " vendor " << infoValue << endl;

       platform.getInfo(CL_PLATFORM_VERSION, &infoValue);
       cout << "Platform  " << platform_index << " version " << infoValue << endl;

       platform.getInfo(CL_PLATFORM_PROFILE, &infoValue);
       cout << "Platform  " << platform_index << " profile " << infoValue << endl;


      /* Find size of extension data */
      //err = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);		
      err = platform.getInfo(CL_PLATFORM_EXTENSIONS, &extInfo);
      if(err < 0) {
         perror("Couldn't read extension data.");			
         exit(1);
      }								

      /* Look for ICD extension */   
      
      if(extInfo.find(icd_ext) != extInfo.npos) {
          cout << "Platform  "<< platform_index  <<" supports the " << icd_ext  <<" extension" << endl;
         return 0;
      }
      

      platform_index++;
   }

   
   cout << "No platforms support the " << icd_ext << endl;
      
   return 0;
} 