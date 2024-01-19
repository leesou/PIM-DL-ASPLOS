set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g -O3 ")

message("CMAKE CXX FLAGS: " ${CMAKE_CXX_FLAGS})

add_executable(tuner auto_tuner.cpp)

target_compile_features(tuner PUBLIC cxx_std_11)
target_link_libraries(tuner yaml-cpp)
