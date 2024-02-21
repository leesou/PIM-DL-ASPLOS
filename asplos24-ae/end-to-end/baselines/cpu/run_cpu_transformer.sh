#!/bin/sh

echo "choose the test type you want to run: full-model, breakdown"
read test_type
full_model=0
if [ "$test_type" == "full-model" ]; then
    full_model=1
fi

echo "choose the inference type you want to run: bert-base, bert-large, vit-huge, other"
read inference_type
bert_base=0
bert_large=0
vit_huge=0
token_number_str=0
batch_size_str=0
model_dim_str=0
head_num_str=0
if [ "$inference_type" == "other" ]; then
    echo "type in the token number per sequence"
    read token_number_str
    echo "type in the batch size"
    read batch_size_str
    echo "type in the model dim"
    read model_dim_str
    echo "type in the head number"
    read head_num_str
elif [ "$inference_type" == "bert-base" ]; then
    bert_base=1
elif [ "$inference_type" == "bert-large" ]; then
    bert_large=1
elif [ "$inference_type" == "vit-huge" ]; then
    vit_huge=1
fi

mkdir -p build
cd build
cmake -DPROFILE_TRANSFORMER=${full_model} -DBERT_BASE=${bert_base} -DBERT_LARGE=${bert_large} -DVIT_HUGE=${vit_huge} \
      ..
make -j
./bin/main $token_number_str $batch_size_str $model_dim_str $head_num_str
