cmake_minimum_required( VERSION 2.6 )
project( openclInAction )


set_property(GLOBAL PROPERTY USE_FOLDERS On)
include_directories( ${PROJECT_SOURCE_DIR}/inc/)
include_directories( "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/")
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64")

#find_package(OpenCV REQUIRED)
#include_directories( ${OpenCV_INCLUDE_DIRS})
SUBDIRS( ch1)		
