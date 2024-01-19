# LUTerize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter, UninitializedParameter
from LUTNeuro.LUTLinear_t import LUTLinear_t

from functools import partial
import os
from typing import List
import transformers


class LUTerize:
    def __init__(self, 
                model, 
                dataloader,
                tokenizer, 
                logger,
                activation_dir, 
                centroid_dir,
                output_dir,
                nsharecodebook=1,
                ncentroid=16,
                vec_len=16, 
                recorded_ops=(nn.Linear),
                nsample=1,
                kmeans_iter=5000,
                weight_transpose=True,
                lut_layers=[],
                block_layer_list=[],
                init_centroids=True,
                fp16=False,
                distance_p="2.0"):
   
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer 
        self.activation_dir = activation_dir
        self.centroid_dir = centroid_dir
        self.output_dir = output_dir
        self.recorded_ops = recorded_ops
        self.nsample = nsample
        self.logger = logger
        self.nsharecodebook = nsharecodebook
        self.ncentroid = ncentroid
        self.vec_len = vec_len
        self.kmeans_iter = kmeans_iter      
        self.weight_transpose = weight_transpose
        self.init_centroids = init_centroids
        
        self.lut_layers = lut_layers 
        self.activations = {}
        self.activation_count = {}
        self.block_layer_list = block_layer_list
        
        self.fp16 = fp16
        self.distance_p = distance_p

    def _hook(self, layer_name, module, input, output):
        key = f"{layer_name}"
        if key not in self.activation_count:
            self.activation_count[key] = 0
        input_np = input[0].cpu().numpy()
        filepath = os.path.join(self.activation_dir, f"{key}_input_{self.activation_count[key]}.npy")
        np.save(filepath, input_np)
        self.activation_count[key] += 1
     
    def _register_hooks(self, module, prefix=""):
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and child_name not in self.block_layer_list:
                self.logger.info(f"Registering hook for {child_name}")
                hook = child.register_forward_hook(partial(self._hook, child_name))
                self.hooks.append(hook)
            else:
                self._register_hooks(child, prefix=child_name)
     
    def save_activations(self):
        self.hooks = []
        self._register_hooks(self.model)
        
        sample_count = 0
        for batch in self.dataloader:
            for key in batch:
                if isinstance(batch, dict):
                    for key in batch:
                        batch[key] = batch[key].to(self.model.device)
                elif isinstance(batch, tuple): 
                    tk_t, lg_t = batch
                    tk_t = tk_t.to(self.model.device)
                    lg_t = lg_t.to(self.model.device)
                    attention_mask = torch.arange(tk_t.size(1), device=self.model.device).expand(tk_t.size(0), -1) < lg_t.unsqueeze(1)
                    batch = {"input_ids": tk_t, "attention_mask": attention_mask}
                elif isinstance(batch, transformers.BatchEncoding):
                    batch = batch.to(self.model.device)
                else:
                    raise ValueError("Unsupported batch type")
             
            with torch.no_grad():
                output = self.model(**batch)
                self.logger.info(f"Model output: {output}")
                
            sample_count += 1
            print(f"Sample count: {sample_count}")
            if sample_count >= self.nsample:
                break

        for hook in self.hooks:
            hook.remove()
      
    def cluster_activations(self):
        collected_files = []
        print(self)
        for filename in os.listdir(self.activation_dir):
            if filename.endswith(".npy") and filename not in collected_files:
                layer_name = filename.split("_")[:-1]
                index = filename.split("_")[-1]
                self.logger.info(f"collect Layer name: {layer_name}")
                collected_files.append(filename)
                
                layer_activations = []
                for i in range(self.nsample):
                    file_path = os.path.join(self.activation_dir, f"{'_'.join(layer_name)}_{i}.npy")
                    activation = np.load(file_path)
                    self.logger.info(f"{layer_name}, {i}, shape: {activation.shape}") 
                    layer_activations.append(activation)
                    collected_files.append(f"{'_'.join(layer_name)}_{i}.npy")
                merged_activation = np.concatenate(layer_activations, axis=-2) 
                self.logger.info(f"merged activation shape: {merged_activation.shape}")
                
                if len(merged_activation.shape) == 2:
                    merged_activation = np.expand_dims(merged_activation, axis=1)
                    self.logger.info(f"expand activation shape: {merged_activation.shape}")
                elif len(merged_activation.shape) == 3:
                    merged_activation = merged_activation
                    self.logger.info(f"expand activation shape: {merged_activation.shape}")
                else:
                    assert(False, "Unsupported activation shape")
                nsample, seq, hidden = merged_activation.shape
                print(f"nsharecodebook: {self.nsharecodebook}, vec_len: {self.vec_len}") 
                assert(hidden % (self.nsharecodebook * self.vec_len) == 0)
                ncodebook = int(hidden / (self.nsharecodebook * self.vec_len))
                shared_hidden = hidden // ncodebook 
                reshaped_codebook_activations = np.zeros((ncodebook, nsample * seq, shared_hidden))
                for i in range(ncodebook):
                    reshaped_codebook_activations[i] = merged_activation.reshape(nsample * seq, hidden)[:, i*shared_hidden:(i+1)*shared_hidden]
                self.logger.info(f"reshaped codebook activations shape: {reshaped_codebook_activations.shape}")
                
                cluster_activations = reshaped_codebook_activations.reshape(ncodebook, -1, self.vec_len)
                self.logger.info(f"reshaped codebook activations shape: {cluster_activations.shape}")
                
                centroids = np.zeros((ncodebook, self.ncentroid, self.vec_len)) 
                for i in range(ncodebook): 
                    codebook_activations = cluster_activations[i].reshape(-1, self.vec_len) 
                    kmeans = KMeans(n_clusters=self.ncentroid, max_iter=self.kmeans_iter)
                    kmeans.fit(codebook_activations.astype(np.float32))
                    self.logger.info(f'{i}, kmeans: lable {kmeans.labels_.shape} centroids: {kmeans.cluster_centers_.shape}')
                    centroids[i] = kmeans.cluster_centers_
                  
                np.save(os.path.join(self.centroid_dir, f"{'_'.join(layer_name)}.npy"), centroids)
                self.logger.info(f'save {layer_name} centroids')

    def luterize_model(self):
        def _replace_linear_with_lutlinear(module: nn.Module, name_parts: List[str] = None):
            if name_parts is None:
                name_parts = []
            for name, child in module.named_children():
                new_name_parts = name_parts + [name]
                if isinstance(child, nn.Linear) and name not in ['pre_classifier', 'classifier_input', 'classifier', 'vocab_transform', 'vocab_projector', 'lm_head']:
                    layer_name = ".".join(new_name_parts)
                    layer_idx = int(layer_name.split(".")[3]) if len(layer_name.split(".")) > 4 else 'dense'
                    if layer_idx in self.lut_layers or self.lut_layers == []:
                        ncodebook = int(child.in_features / (self.nsharecodebook * self.vec_len))
                        if self.weight_transpose:
                            lut_linear = LUTLinear_t( in_features=child.in_features, 
                                                    out_features=child.out_features,
                                                    dtype=child.weight.dtype,
                                                    ncentroids=self.ncentroid,
                                                    vec_len=self.vec_len, 
                                                    bias=child.bias is not None,
                                                    fp16=self.fp16,
                                                    distance_p=self.distance_p,)
                            lut_linear.weight.data.copy_(child.weight.data.transpose(0, 1).clone())
                        else:
                            lut_linear = LUTLinear( in_features=child.in_features, 
                                                    out_features=child.out_features,
                                                    dtype=child.weight.dtype,
                                                    ncodebooks=ncodebook,
                                                    ncentroids=self.ncentroid,
                                                    vec_len=self.vec_len, 
                                                    bias=child.bias is not None,
                                                    fp16=self.fp16)
                            lut_linear.weight.data.copy_(child.weight.data)
                        
                        if self.init_centroids:                        
                            self.logger.info(f"Replace {layer_name}, {name} with LUTLinear")
                            filepath = os.path.join(self.centroid_dir, f"{layer_name}_input.npy")
                            self.logger.info(f"Load centroids from {filepath}")
                            centroids = torch.from_numpy(np.load(filepath)).type(child.weight.dtype)
                            
                            self.logger.info(f"Copy centroids from {filepath} to LUTLinear")
                            lut_linear.centroids.data.copy_(centroids)
                        
                        if child.bias is not None:
                            lut_linear.bias.data.copy_(child.bias.data) 
                        setattr(module, name, lut_linear)
                else:
                    _replace_linear_with_lutlinear(child, new_name_parts)
        
        _replace_linear_with_lutlinear(self.model) 
