#pragma once
#include <string>
extern "C"{
    #include <dpu.h>
}

#ifndef DPU_BINARY
    #define DPU_BINARY "./build/bin/dpu_bin"
#endif

static uint32_t allocate_dpu(dpu_set_t* dpu_set, uint32_t dpu_num)
{
    uint32_t allocated_dpu_num;
    DPU_ASSERT(dpu_alloc(dpu_num, NULL, dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(*dpu_set, &allocated_dpu_num));
#ifdef DEBUG
    printf("[DPU_INFO]: NR ALLOCATED DPUS: %d\n", allocated_dpu_num);
#endif
    return allocated_dpu_num;
}

static uint32_t allocate_rank(dpu_set_t* dpu_set, uint32_t rank_num)
{
    uint32_t allocated_dpu_num;
    DPU_ASSERT(dpu_alloc_ranks(rank_num, NULL, dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(*dpu_set, &allocated_dpu_num));
#ifdef DEBUG
    printf("[DPU_INFO]: NR ALLOCATED DPUS: %d\n", allocated_dpu_num);
#endif
    return allocated_dpu_num;
}

static void free_dpu_set(dpu_set_t* dpu_set)
{
    DPU_ASSERT(dpu_free(*dpu_set));
}

static void load_binary(dpu_set_t* dpu_set, std::string binary_name="")
{
    if(binary_name != "")
    {
#ifdef DEBUG
        printf("[DPU_INFO]: Load binary %s to DPUs\n", binary_name.c_str());
#endif
        DPU_ASSERT(dpu_load(*dpu_set, binary_name.c_str(), NULL));
    }
    else
    {
#ifdef DEBUG
        printf("[DPU_INFO]: Empty binary input, load default binary string %s to DPUs\n", DPU_BINARY);
#endif
        DPU_ASSERT(dpu_load(*dpu_set, DPU_BINARY, NULL));
    }
}
