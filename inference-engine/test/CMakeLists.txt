set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g -O3 ")



add_executable(test_amm test_amm.cpp)
target_link_libraries(test_amm host ggml)
target_compile_definitions(test_amm PUBLIC NR_DPUS_PER_RANK=${NR_DPUS_PER_RANK})



add_executable(test_transformer_layer test_transformer_layer.cpp)
target_link_libraries(test_transformer_layer host ggml)
target_compile_definitions(test_transformer_layer PUBLIC NR_DPUS_PER_RANK=${NR_DPUS_PER_RANK})
if(MEASURE_ENERGY)
    target_compile_definitions(test_transformer_layer PUBLIC MEASURE_ENERGY)
endif()
if(SEPARATE)
    target_compile_definitions(test_transformer_layer PUBLIC SEPARATE)
endif()



add_executable(test_transformer test_transformer.cpp)
target_link_libraries(test_transformer host ggml cnpy)
target_compile_definitions(test_transformer PUBLIC NR_DPUS_PER_RANK=${NR_DPUS_PER_RANK})
if(SEPARATE)
    target_compile_definitions(test_transformer PUBLIC SEPARATE)
endif()
