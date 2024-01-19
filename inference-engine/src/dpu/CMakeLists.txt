set(CMAKE_C_COMPILER "dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNR_TASKLETS=${NR_TASKLETS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
if(DEBUG_MODE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG")
endif()


message("CMAKE C FLAGS: " ${CMAKE_C_FLAGS})




if(DYNAMIC)
add_executable(dpu_bin_dynamic pim_lut_kernel.c)
target_link_libraries(dpu_bin_dynamic ${DPU_LIBRARY_DIRS})
# cb, ct shape
target_compile_definitions(dpu_bin_dynamic PRIVATE NUM_CENTROID=${NUM_CENTROID_D})
target_compile_definitions(dpu_bin_dynamic PRIVATE NUM_CODEBOOK=${NUM_CODEBOOK_D})
# loop order and load type
if(LOOP_ORDER_D EQUAL "0")
target_compile_definitions(dpu_bin_dynamic PRIVATE LOOP_ORDER_NFC)
elseif(LOOP_ORDER_D EQUAL "1")
target_compile_definitions(dpu_bin_dynamic PRIVATE LOOP_ORDER_NCF)
elseif(LOOP_ORDER_D EQUAL "2")
target_compile_definitions(dpu_bin_dynamic PRIVATE LOOP_ORDER_FNC)
elseif(LOOP_ORDER_D EQUAL "3")
target_compile_definitions(dpu_bin_dynamic PRIVATE LOOP_ORDER_FCN)
elseif(LOOP_ORDER_D EQUAL "4")
target_compile_definitions(dpu_bin_dynamic PRIVATE LOOP_ORDER_CNF)
elseif(LOOP_ORDER_D EQUAL "5")
target_compile_definitions(dpu_bin_dynamic PRIVATE LOOP_ORDER_CFN)
endif()
if(LUT_LOAD_TYPE_D EQUAL "0")
target_compile_definitions(dpu_bin_dynamic PRIVATE STATIC_LUT_TABLE)
elseif(LUT_LOAD_TYPE_D EQUAL "1")
target_compile_definitions(dpu_bin_dynamic PRIVATE FINE_GRAIN)
endif()
# stile sizes
target_compile_definitions(dpu_bin_dynamic PRIVATE N_STILE_SIZE=${N_STILE_SIZE_D})
target_compile_definitions(dpu_bin_dynamic PRIVATE FEATURE_STILE_SIZE=${FEATURE_STILE_SIZE_D})
# mtile sizes
target_compile_definitions(dpu_bin_dynamic PRIVATE N_MTILE_SIZE=${N_MTILE_SIZE_D})
target_compile_definitions(dpu_bin_dynamic PRIVATE FEATURE_MTILE_SIZE=${FEATURE_MTILE_SIZE_D})
target_compile_definitions(dpu_bin_dynamic PRIVATE CB_MTILE_SIZE=${CB_MTILE_SIZE_D})
# load tile sizes
target_compile_definitions(dpu_bin_dynamic PRIVATE FEATURE_LOAD_TILE_SIZE=${FEATURE_LOAD_TILE_SIZE_D})
target_compile_definitions(dpu_bin_dynamic PRIVATE CB_LOAD_TILE_SIZE=${CB_LOAD_TILE_SIZE_D})
endif()




if(TRANSFORMER)
############################# QKV LUT
add_executable(dpu_bin_qkv pim_lut_kernel.c)
target_link_libraries(dpu_bin_qkv ${DPU_LIBRARY_DIRS})
# cb, ct shape
target_compile_definitions(dpu_bin_qkv PRIVATE NUM_CENTROID=${NUM_CENTROID_QKV})
target_compile_definitions(dpu_bin_qkv PRIVATE NUM_CODEBOOK=${NUM_CODEBOOK_QKV})
# loop order and load type
if(LOOP_ORDER_QKV EQUAL "0")
target_compile_definitions(dpu_bin_qkv PRIVATE LOOP_ORDER_NFC)
elseif(LOOP_ORDER_QKV EQUAL "1")
target_compile_definitions(dpu_bin_qkv PRIVATE LOOP_ORDER_NCF)
elseif(LOOP_ORDER_QKV EQUAL "2")
target_compile_definitions(dpu_bin_qkv PRIVATE LOOP_ORDER_FNC)
elseif(LOOP_ORDER_QKV EQUAL "3")
target_compile_definitions(dpu_bin_qkv PRIVATE LOOP_ORDER_FCN)
elseif(LOOP_ORDER_QKV EQUAL "4")
target_compile_definitions(dpu_bin_qkv PRIVATE LOOP_ORDER_CNF)
elseif(LOOP_ORDER_QKV EQUAL "5")
target_compile_definitions(dpu_bin_qkv PRIVATE LOOP_ORDER_CFN)
endif()
if(LUT_LOAD_TYPE_QKV EQUAL "0")
target_compile_definitions(dpu_bin_qkv PRIVATE STATIC_LUT_TABLE)
elseif(LUT_LOAD_TYPE_QKV EQUAL "1")
target_compile_definitions(dpu_bin_qkv PRIVATE FINE_GRAIN)
endif()
# stile sizes
target_compile_definitions(dpu_bin_qkv PRIVATE N_STILE_SIZE=${N_STILE_SIZE_QKV})
target_compile_definitions(dpu_bin_qkv PRIVATE FEATURE_STILE_SIZE=${FEATURE_STILE_SIZE_QKV})
# mtile sizes
target_compile_definitions(dpu_bin_qkv PRIVATE N_MTILE_SIZE=${N_MTILE_SIZE_QKV})
target_compile_definitions(dpu_bin_qkv PRIVATE FEATURE_MTILE_SIZE=${FEATURE_MTILE_SIZE_QKV})
target_compile_definitions(dpu_bin_qkv PRIVATE CB_MTILE_SIZE=${CB_MTILE_SIZE_QKV})
# load tile sizes
target_compile_definitions(dpu_bin_qkv PRIVATE FEATURE_LOAD_TILE_SIZE=${FEATURE_LOAD_TILE_SIZE_QKV})
target_compile_definitions(dpu_bin_qkv PRIVATE CB_LOAD_TILE_SIZE=${CB_LOAD_TILE_SIZE_QKV})

############################# O LUT
add_executable(dpu_bin_o pim_lut_kernel.c)
target_link_libraries(dpu_bin_o ${DPU_LIBRARY_DIRS})
# cb, ct shape
target_compile_definitions(dpu_bin_o PRIVATE NUM_CENTROID=${NUM_CENTROID_QKV})
target_compile_definitions(dpu_bin_o PRIVATE NUM_CODEBOOK=${NUM_CODEBOOK_QKV})
# loop order and load type
if(LOOP_ORDER_QKV EQUAL "0")
target_compile_definitions(dpu_bin_o PRIVATE LOOP_ORDER_NFC)
elseif(LOOP_ORDER_QKV EQUAL "1")
target_compile_definitions(dpu_bin_o PRIVATE LOOP_ORDER_NCF)
elseif(LOOP_ORDER_QKV EQUAL "2")
target_compile_definitions(dpu_bin_o PRIVATE LOOP_ORDER_FNC)
elseif(LOOP_ORDER_QKV EQUAL "3")
target_compile_definitions(dpu_bin_o PRIVATE LOOP_ORDER_FCN)
elseif(LOOP_ORDER_QKV EQUAL "4")
target_compile_definitions(dpu_bin_o PRIVATE LOOP_ORDER_CNF)
elseif(LOOP_ORDER_QKV EQUAL "5")
target_compile_definitions(dpu_bin_o PRIVATE LOOP_ORDER_CFN)
endif()
if(LUT_LOAD_TYPE_QKV EQUAL "0")
target_compile_definitions(dpu_bin_o PRIVATE STATIC_LUT_TABLE)
elseif(LUT_LOAD_TYPE_QKV EQUAL "1")
target_compile_definitions(dpu_bin_o PRIVATE FINE_GRAIN)
endif()
# stile sizes
target_compile_definitions(dpu_bin_o PRIVATE N_STILE_SIZE=${N_STILE_SIZE_QKV})
target_compile_definitions(dpu_bin_o PRIVATE FEATURE_STILE_SIZE=${FEATURE_STILE_SIZE_QKV})
# mtile sizes
target_compile_definitions(dpu_bin_o PRIVATE N_MTILE_SIZE=${N_MTILE_SIZE_QKV})
target_compile_definitions(dpu_bin_o PRIVATE FEATURE_MTILE_SIZE=${FEATURE_MTILE_SIZE_QKV})
target_compile_definitions(dpu_bin_o PRIVATE CB_MTILE_SIZE=${CB_MTILE_SIZE_QKV})
# load tile sizes
target_compile_definitions(dpu_bin_o PRIVATE FEATURE_LOAD_TILE_SIZE=${FEATURE_LOAD_TILE_SIZE_QKV})
target_compile_definitions(dpu_bin_o PRIVATE CB_LOAD_TILE_SIZE=${CB_LOAD_TILE_SIZE_QKV})

############################# FFN1 LUT
add_executable(dpu_bin_ffn1 pim_lut_kernel.c)
target_link_libraries(dpu_bin_ffn1 ${DPU_LIBRARY_DIRS})
# cb, ct shape
target_compile_definitions(dpu_bin_ffn1 PRIVATE NUM_CENTROID=${NUM_CENTROID_FFN1})
target_compile_definitions(dpu_bin_ffn1 PRIVATE NUM_CODEBOOK=${NUM_CODEBOOK_FFN1})
# loop order and load type
if(LOOP_ORDER_FFN1 EQUAL "0")
target_compile_definitions(dpu_bin_ffn1 PRIVATE LOOP_ORDER_NFC)
elseif(LOOP_ORDER_FFN1 EQUAL "1")
target_compile_definitions(dpu_bin_ffn1 PRIVATE LOOP_ORDER_NCF)
elseif(LOOP_ORDER_FFN1 EQUAL "2")
target_compile_definitions(dpu_bin_ffn1 PRIVATE LOOP_ORDER_FNC)
elseif(LOOP_ORDER_FFN1 EQUAL "3")
target_compile_definitions(dpu_bin_ffn1 PRIVATE LOOP_ORDER_FCN)
elseif(LOOP_ORDER_FFN1 EQUAL "4")
target_compile_definitions(dpu_bin_ffn1 PRIVATE LOOP_ORDER_CNF)
elseif(LOOP_ORDER_FFN1 EQUAL "5")
target_compile_definitions(dpu_bin_ffn1 PRIVATE LOOP_ORDER_CFN)
endif()
if(LUT_LOAD_TYPE_FFN1 EQUAL "0")
target_compile_definitions(dpu_bin_ffn1 PRIVATE STATIC_LUT_TABLE)
elseif(LUT_LOAD_TYPE_FFN1 EQUAL "1")
target_compile_definitions(dpu_bin_ffn1 PRIVATE FINE_GRAIN)
endif()
# stile sizes
target_compile_definitions(dpu_bin_ffn1 PRIVATE N_STILE_SIZE=${N_STILE_SIZE_FFN1})
target_compile_definitions(dpu_bin_ffn1 PRIVATE FEATURE_STILE_SIZE=${FEATURE_STILE_SIZE_FFN1})
# mtile sizes
target_compile_definitions(dpu_bin_ffn1 PRIVATE N_MTILE_SIZE=${N_MTILE_SIZE_FFN1})
target_compile_definitions(dpu_bin_ffn1 PRIVATE FEATURE_MTILE_SIZE=${FEATURE_MTILE_SIZE_FFN1})
target_compile_definitions(dpu_bin_ffn1 PRIVATE CB_MTILE_SIZE=${CB_MTILE_SIZE_FFN1})
# load tile sizes
target_compile_definitions(dpu_bin_ffn1 PRIVATE FEATURE_LOAD_TILE_SIZE=${FEATURE_LOAD_TILE_SIZE_FFN1})
target_compile_definitions(dpu_bin_ffn1 PRIVATE CB_LOAD_TILE_SIZE=${CB_LOAD_TILE_SIZE_FFN1})

############################# FFN2 LUT
add_executable(dpu_bin_ffn2 pim_lut_kernel.c)
target_link_libraries(dpu_bin_ffn2 ${DPU_LIBRARY_DIRS})
# cb, ct shape
target_compile_definitions(dpu_bin_ffn2 PRIVATE NUM_CENTROID=${NUM_CENTROID_FFN2})
target_compile_definitions(dpu_bin_ffn2 PRIVATE NUM_CODEBOOK=${NUM_CODEBOOK_FFN2})
# loop order and load type
if(LOOP_ORDER_FFN2 EQUAL "0")
target_compile_definitions(dpu_bin_ffn2 PRIVATE LOOP_ORDER_NFC)
elseif(LOOP_ORDER_FFN2 EQUAL "1")
target_compile_definitions(dpu_bin_ffn2 PRIVATE LOOP_ORDER_NCF)
elseif(LOOP_ORDER_FFN2 EQUAL "2")
target_compile_definitions(dpu_bin_ffn2 PRIVATE LOOP_ORDER_FNC)
elseif(LOOP_ORDER_FFN2 EQUAL "3")
target_compile_definitions(dpu_bin_ffn2 PRIVATE LOOP_ORDER_FCN)
elseif(LOOP_ORDER_FFN2 EQUAL "4")
target_compile_definitions(dpu_bin_ffn2 PRIVATE LOOP_ORDER_CNF)
elseif(LOOP_ORDER_FFN2 EQUAL "5")
target_compile_definitions(dpu_bin_ffn2 PRIVATE LOOP_ORDER_CFN)
endif()
if(LUT_LOAD_TYPE_FFN2 EQUAL "0")
target_compile_definitions(dpu_bin_ffn2 PRIVATE STATIC_LUT_TABLE)
elseif(LUT_LOAD_TYPE_FFN2 EQUAL "1")
target_compile_definitions(dpu_bin_ffn2 PRIVATE FINE_GRAIN)
endif()
# stile sizes
target_compile_definitions(dpu_bin_ffn2 PRIVATE N_STILE_SIZE=${N_STILE_SIZE_FFN2})
target_compile_definitions(dpu_bin_ffn2 PRIVATE FEATURE_STILE_SIZE=${FEATURE_STILE_SIZE_FFN2})
# mtile sizes
target_compile_definitions(dpu_bin_ffn2 PRIVATE N_MTILE_SIZE=${N_MTILE_SIZE_FFN2})
target_compile_definitions(dpu_bin_ffn2 PRIVATE FEATURE_MTILE_SIZE=${FEATURE_MTILE_SIZE_FFN2})
target_compile_definitions(dpu_bin_ffn2 PRIVATE CB_MTILE_SIZE=${CB_MTILE_SIZE_FFN2})
# load tile sizes
target_compile_definitions(dpu_bin_ffn2 PRIVATE FEATURE_LOAD_TILE_SIZE=${FEATURE_LOAD_TILE_SIZE_FFN2})
target_compile_definitions(dpu_bin_ffn2 PRIVATE CB_LOAD_TILE_SIZE=${CB_LOAD_TILE_SIZE_FFN2})
endif()

