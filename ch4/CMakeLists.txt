#指定生成目标
add_executable(float_config float_config.c)
target_link_libraries(float_config OpenCL )                                                                             
set_target_properties(float_config PROPERTIES FOLDER "chapter4") 


#指定生成目标
add_executable(vector_widths vector_widths.c)
target_link_libraries(vector_widths OpenCL )                                                                             
set_target_properties(vector_widths PROPERTIES FOLDER "chapter4") 


