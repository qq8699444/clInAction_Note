add_executable(opTest opTest.cpp)
target_link_libraries(opTest OpenCL )                                                                             
set_target_properties(opTest PROPERTIES FOLDER "chapter5") 

add_executable(idCheck idCheck.cpp)
target_link_libraries(idCheck OpenCL )                                                                             
set_target_properties(idCheck PROPERTIES FOLDER "chapter5") 

