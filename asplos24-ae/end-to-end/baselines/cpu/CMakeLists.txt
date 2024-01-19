cmake_minimum_required(VERSION 3.12)
project("transformer_baseline" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#
# Compile flags
#

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_compile_options(-Wno-format)

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (BERT_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (BERT_GPROF)
        add_compile_options(-pg)
    endif()
    if (BERT_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-mcpu=native)
        endif()
        # TODO: armv6,7,8 version specific flags
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (BERT_AVX2)
            add_compile_options(/arch:AVX2)
        elseif (BERT_AVX)
            add_compile_options(/arch:AVX)
        endif()
    else()
        add_compile_options(-mf16c)
        if (BERT_FMA)
            add_compile_options(-mfma)
        endif()
        if (BERT_AVX)
            add_compile_options(-mavx)
        endif()
        if (BERT_AVX2)
            add_compile_options(-mavx2)
        endif()
    endif()
else()
    # TODO: support PowerPC
    message(STATUS "Unknown architecture")
endif()

# OPENMP
FIND_PACKAGE(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#
# Build libraries
#

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

add_subdirectory(ggml)

add_library(transformer
            transformer.cpp
            transformer.h)

target_include_directories(transformer PUBLIC .)
# target_compile_features(transformer PUBLIC cxx_std_20)
target_link_libraries(transformer PRIVATE ggml ${BERT_EXTRA_LIBS})

#
# Build binaries
#

set(PROFILE_TRANSFORMER TRUE)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE transformer ggml)
if(PROFILE_TRANSFORMER)
target_compile_definitions(main PUBLIC PROFILE_TRANSFORMER)
endif()

