add_executable(callback callback.cpp )
target_link_libraries(callback OpenCL)                                                                             
set_target_properties(callback PROPERTIES FOLDER "chapter7") 


add_executable(userEvent userEvent.cpp )
target_link_libraries(userEvent OpenCL)                                                                             
set_target_properties(userEvent PROPERTIES FOLDER "chapter7") 

add_executable(profileRead profileRead.cpp )
target_link_libraries(profileRead OpenCL)                                                                             
set_target_properties(profileRead PROPERTIES FOLDER "chapter7") 

add_executable(profileItem profileItem.cpp )
target_link_libraries(profileItem OpenCL)                                                                             
set_target_properties(profileItem PROPERTIES FOLDER "chapter7") 


add_executable(atomic atomic.cpp)
target_link_libraries(atomic OpenCL)                                                                             
set_target_properties(atomic PROPERTIES FOLDER "chapter7") 