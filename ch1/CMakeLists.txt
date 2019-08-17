#指定生成目标
add_executable(matvec matvec.c)
#链接共享库
target_link_libraries(matvec OpenCL )                                                                             
set_target_properties(matvec PROPERTIES FOLDER "chapter1") 
