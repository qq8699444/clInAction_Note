cmake_minimum_required( VERSION 2.6 )
project( openclInAction )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set_property(GLOBAL PROPERTY USE_FOLDERS On)
include_directories( ${PROJECT_SOURCE_DIR}/inc/)
include_directories( "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/")
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64")

set(OpenCV_DIR E:/opencv/opencv-410-build/install)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})


SUBDIRS( ch1 ch2 ch3 ch4 ch5 ch6)		
