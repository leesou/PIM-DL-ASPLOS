CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
project(pim_gemm)

############################### common parameters

# DPU setup
set(NR_TASKLETS 16)
set(NR_DPUS_PER_RANK 64)

############################### GEMM kernel test setup

# parallelism
set(INPUT_PARALLELISM_GEMM 128)
set(LUT_PARALLELISM_GEMM 8)
set(DPU_NUM_GEMM ${INPUT_PARALLELISM_GEMM}*${LUT_PARALLELISM_GEMM})
# tile size
set(N_STILE_SIZE_GEMM 256)
set(FEATURE_STILE_SIZE_GEMM 160)
# on-chip tile size
set(N_MTILE_SIZE_GEMM 16)
set(HIDDEN_MTILE_SIZE_GEMM 256)
set(FEATURE_MTILE_SIZE_GEMM 16)
# shape
set(N_GEMM ${INPUT_PARALLELISM_GEMM}*${N_STILE_SIZE_GEMM})
set(HIDDEN_DIM_GEMM 5120)
set(FEATURE_LEN_GEMM ${LUT_PARALLELISM_GEMM}*${FEATURE_STILE_SIZE_GEMM})

############################### library options

#DPU libiraries
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

# MKL
find_package(MKL REQUIRED)
if(MKL_FOUND)
    message("MKL FOUND")
    include_directories(${MKL_INCLUDE})
    link_directories(${MKL_ROOT}/lib/intel64)
    set(MKL_LINKER_FLAGS "${MKL_LINKER_FLAGS} ${MKL_LINKER_FLAGS_OTHER}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MKL_LINKER_FLAGS}")
endif()

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

############################### binaries

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g -O3 ")
message("CMAKE CXX FLAGS: " ${CMAKE_CXX_FLAGS})

set(CMAKE_C_COMPILER "dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNR_TASKLETS=${NR_TASKLETS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
if(DEBUG_MODE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG")
endif()
message("CMAKE C FLAGS: " ${CMAKE_C_FLAGS})

add_executable(dpu_bin_gemm gemm_kernel.c)
target_link_libraries(dpu_bin_gemm ${DPU_LIBRARY_DIRS})
target_compile_definitions(dpu_bin_gemm PRIVATE N_STILE_SIZE=${N_STILE_SIZE_GEMM})
target_compile_definitions(dpu_bin_gemm PRIVATE FEATURE_STILE_SIZE=${FEATURE_STILE_SIZE_GEMM})
target_compile_definitions(dpu_bin_gemm PRIVATE HIDDEN_DIM=${HIDDEN_DIM_GEMM})
target_compile_definitions(dpu_bin_gemm PRIVATE N_MTILE_SIZE=${N_MTILE_SIZE_GEMM})
target_compile_definitions(dpu_bin_gemm PRIVATE HIDDEN_MTILE_SIZE=${HIDDEN_MTILE_SIZE_GEMM})
target_compile_definitions(dpu_bin_gemm PRIVATE FEATURE_MTILE_SIZE=${FEATURE_MTILE_SIZE_GEMM})

add_executable(test_gemm test_pim_gemm.cpp)
target_link_libraries(test_gemm ${DPU_LIBRARIES})
target_compile_definitions(test_gemm PUBLIC DPU_NUM=${DPU_NUM_GEMM})
target_compile_definitions(test_gemm PUBLIC NR_DPUS_PER_RANK=${NR_DPUS_PER_RANK})
target_compile_definitions(test_gemm PUBLIC ROW=${N_GEMM})
target_compile_definitions(test_gemm PUBLIC HIDDEN_DIM=${HIDDEN_DIM_GEMM})
target_compile_definitions(test_gemm PUBLIC FEATURE_LEN=${FEATURE_LEN_GEMM})
target_compile_definitions(test_gemm PUBLIC N_STILE_SIZE=${N_STILE_SIZE_GEMM})
target_compile_definitions(test_gemm PUBLIC FEATURE_STILE_SIZE=${FEATURE_STILE_SIZE_GEMM})
target_compile_definitions(test_gemm PUBLIC INPUT_PARALLELISM=${INPUT_PARALLELISM_GEMM})
target_compile_definitions(test_gemm PUBLIC LUT_PARALLELISM=${LUT_PARALLELISM_GEMM})
