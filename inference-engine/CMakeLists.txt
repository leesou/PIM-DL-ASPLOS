CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
project(pim_lut_nn)

# DPU setup
set(NR_TASKLETS 16)
set(NR_DPUS_PER_RANK 64)

if(MEASURE_ENERGY)
    set(MEASURE_ENERGY TRUE)
else()
    set(MEASURE_ENERGY FALSE)
endif()

if(SEPARATE)
    set(SEPARATE TRUE)
else()
    set(SEPARATE FALSE)
endif()

if(LATENCY_BREAKDOWN)
    set(AMM_BREAKDOWN TRUE)
else()
    set(AMM_BREAKDOWN FALSE)
endif()

if(LUT_BREAKDOWN)
    set(TRANSFORMER_BREAKDOWN TRUE)
else()
    set(TRANSFORMER_BREAKDOWN FALSE)
endif()

if(LAYER_BREAKDOWN)
    set(TRANSFORMER_BREAKDOWN_AMM TRUE)
else()
    set(TRANSFORMER_BREAKDOWN_AMM FALSE)
endif()

set(DEBUG_MODE FALSE)

# DPU libiraries
find_package(PkgConfig REQUIRED)
pkg_search_module(DPU REQUIRED dpu)
include_directories(${DPU_INCLUDE_DIRS})

# OPENMP
FIND_PACKAGE(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# for ggml
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC ")

include_directories(3rd/yaml/include)
include_directories(3rd/ggml/include)
include_directories(3rd/cnpy)
include_directories(src/dpu)
include_directories(src/host)
include_directories(src/tuner)
include_directories(test)

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 3rd packages
add_subdirectory(3rd)
# main source files
add_subdirectory(src)
# test files
add_subdirectory(test)
