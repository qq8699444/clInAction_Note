add_executable(simpleImg simpleImg.cpp imgShow.cpp)
target_link_libraries(simpleImg OpenCL ${OpenCV_LIBS})                                                                             
set_target_properties(simpleImg PROPERTIES FOLDER "chapter6") 

add_executable(scaleImg scaleImg.cpp imgShow.cpp)
target_link_libraries(scaleImg OpenCL ${OpenCV_LIBS})                                                                             
set_target_properties(scaleImg PROPERTIES FOLDER "chapter6") 

