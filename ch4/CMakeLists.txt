add_executable(hello_kernel hello_kernel.cpp)
target_link_libraries(hello_kernel OpenCL )                                                                             
set_target_properties(hello_kernel PROPERTIES FOLDER "chapter4") 

add_executable(float_config float_config.cpp)
target_link_libraries(float_config OpenCL )                                                                             
set_target_properties(float_config PROPERTIES FOLDER "chapter4") 



add_executable(vector_widths vector_widths.cpp)
target_link_libraries(vector_widths OpenCL )                                                                             
set_target_properties(vector_widths PROPERTIES FOLDER "chapter4") 


add_executable(double_test double_test.cpp)
target_link_libraries(double_test OpenCL )                                                                             
set_target_properties(double_test PROPERTIES FOLDER "chapter4") 


add_executable(vector_bytes vector_bytes.cpp)
target_link_libraries(vector_bytes OpenCL )                                                                             
set_target_properties(vector_bytes PROPERTIES FOLDER "chapter4") 