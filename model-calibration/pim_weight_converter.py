import torch
import re
import argparse
import numpy as np
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_input_file', type=str)
    parser.add_argument('--features', type=int, default=768)
    parser.add_argument('--ncentroids', type=int, default=16)
    parser.add_argument('--vec_len', type=int, default=2)
    parser.add_argument('--nsharecodebook', type=int, default=1)
    parser.add_argument('--weight_output_file', type=str)

    args = parser.parse_args()
    return args


def lut_quant(lut):
    # Quantize weight to -128 ~ 127
    # same as original LUT-NN model
    lut_min = lut.min()
    lut_max = lut.max()
    max_abs = torch.maximum(torch.abs(lut_min), torch.abs(lut_max))
    zero_point = torch.zeros_like(lut_min).to(torch.int8)
    scale = max_abs / (127 - zero_point.to(torch.float32))
    lut_quantized = torch.clamp(
                            lut / scale + zero_point,
                            torch.tensor(-128).to(lut.device),
                            torch.tensor(127).to(lut.device)
                        ).round().to(torch.int8)
    bias = -scale*zero_point
    return lut_quantized, scale, bias


def read_weight(args):
    layer_tensors = {}
    layer_num = 0
    with safe_open(args.weight_input_file, framework="pt", device=0) as f:
        for k in f.keys():
            split_k = k.split('.')
            if 'embeddings' in split_k:
                pass
            elif 'layer' in split_k:
                layer_id = split_k[3]
                layer_num = max(layer_num, int(layer_id)+1)
                if layer_id not in layer_tensors.keys():
                    layer_tensors[layer_id] = {}
                k_short = re.sub('.*'+str(layer_id), str(layer_id), k, count=1)
                layer_tensors[layer_id][k_short] = f.get_tensor(k)
            else:
                pass

    for i in range(layer_num):
        tmp_layer_tensors = layer_tensors[str(i)]

        weight_tensors = {}
        centroid_tensors = {}
        for tensor_name in tmp_layer_tensors.keys():
            if '.centroids.weight' in tensor_name:
                op_name = tensor_name.replace('.centroids.weight', '')
                centroid_tensors[op_name] = tmp_layer_tensors[tensor_name]
            elif '.weight' in tensor_name and 'LayerNorm' not in tensor_name:
                op_name = tensor_name.replace('.weight', '')
                weight_tensors[op_name] = tmp_layer_tensors[tensor_name]
            
        lut_tensors = {}
        for op_name in weight_tensors.keys():
            tmp_op_weight = weight_tensors[op_name]
            tmp_op_centroids = centroid_tensors[op_name]

            if 'attention' in op_name or 'intermediate' in op_name:
                in_features = args.features
            else:
                in_features = args.features * 4
            if 'attention' in op_name or 'output' in op_name:
                out_features =args.features
            else:
                out_features = args.features * 4
            ncodebooks = in_features // args.vec_len // args.nsharecodebook
            vec_len = args.vec_len
            ncentroids = args.ncentroids

            weight_flat = tmp_op_weight.view(ncodebooks, vec_len, out_features)
            centroids_flat = tmp_op_centroids.view(ncodebooks, ncentroids, vec_len)
            lut = torch.bmm(centroids_flat, weight_flat)
            lut = lut.view(-1, out_features)
            lut_quantized, scale, bias = lut_quant(lut)
            lut_tensors[op_name] = (lut_quantized, scale, bias)

            tmp_layer_tensors.pop(op_name+'.weight')
            tmp_layer_tensors[op_name+'.lut.tensor'] = lut_quantized
            tmp_layer_tensors[op_name+'.lut.scale'] = scale
            tmp_layer_tensors[op_name+'.lut.bias'] = bias

    return layer_tensors


def store_weight(args, weights):
    tensor_dict = {}
    for layer_id in weights.keys():
        tmp_layer_tensors = weights[layer_id]
        tensor_dict.update(tmp_layer_tensors)
    
    for name in tensor_dict:
        tensor_dict[name] = tensor_dict[name].cpu().numpy()

    np.savez(args.weight_output_file, **tensor_dict)

    loaded_data = np.load(args.weight_output_file)
    for name in tensor_dict:
        if name not in loaded_data.files:
            print('warning, %s not exist in dumped file' % name)
        elif not np.array_equal(tensor_dict[name], loaded_data[name]):
            print('warning, %s not equal in dumped file' % name)
    print('dump finished')


def main():
    args = parse_args()
    model_weight = read_weight(args)
    store_weight(args, model_weight)


if __name__ == '__main__':
    main()
    