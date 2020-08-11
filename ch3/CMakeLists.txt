#指定生成目标
add_executable(buffer_check buffer_check.cpp)
target_link_libraries(buffer_check OpenCL )                                                                             
set_target_properties(buffer_check PROPERTIES FOLDER "chapter3") 


#指定生成目标
add_executable(buffer_test buffer_test.cpp)
target_link_libraries(buffer_test OpenCL )                                                                             
set_target_properties(buffer_test PROPERTIES FOLDER "chapter3") 

#指定生成目标
add_executable(map_copy map_copy.cpp)
target_link_libraries(map_copy OpenCL )                                                                             
set_target_properties(map_copy PROPERTIES FOLDER "chapter3") 


