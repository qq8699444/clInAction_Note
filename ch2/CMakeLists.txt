## platform_ext_test
add_executable(platform_ext_test platform_ext_test.c)
target_link_libraries(platform_ext_test OpenCL )                                                                             
set_target_properties(platform_ext_test PROPERTIES FOLDER "chapter2") 


## device_ext_test
add_executable(device_ext_test device_ext_test.c)
target_link_libraries(device_ext_test OpenCL )                                                                             
set_target_properties(device_ext_test PROPERTIES FOLDER "chapter2") 


add_executable(context_count context_count.c)
target_link_libraries(context_count OpenCL )                                                                             
set_target_properties(context_count PROPERTIES FOLDER "chapter2") 

add_executable(program_build program_build.c)
target_link_libraries(program_build OpenCL )                                                                             
set_target_properties(program_build PROPERTIES FOLDER "chapter2") 

add_executable(kernel_search kernel_search.c)
target_link_libraries(kernel_search OpenCL )                                                                             
set_target_properties(kernel_search PROPERTIES FOLDER "chapter2") 


add_executable(queue_kernel queue_kernel.c)
target_link_libraries(queue_kernel OpenCL )                                                                             
set_target_properties(queue_kernel PROPERTIES FOLDER "chapter2") 
