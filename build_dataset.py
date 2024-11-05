import torch
import os
import transformers
import peft
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForSequenceClassification
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


# build_dataset.py

def stack_vectors(vectors):
    return torch.stack(vectors)


def flatten_and_merge_weights_and_log_positions(weight_dicts):
    vectors = []
    positions = {}  # To store start positions of each key for all dictionaries
    for key_weight_dict in sorted(weight_dicts.keys()): # Sort the keys to ensure consistent ordering
        weight_dict = weight_dicts[key_weight_dict]
        flattened_weights = []
        position = {}
        current_pos = 0

        for key in sorted(weight_dict.keys()): # Sort the keys to ensure consistent ordering
            flat_weight = weight_dict[key].flatten()
            flattened_weights.append(flat_weight)

            position[key] = current_pos
            current_pos += flat_weight.numel()

        concatenated_vector = torch.cat(flattened_weights)
        vectors.append(concatenated_vector)
        positions[key_weight_dict] = position
    
    matrix = stack_vectors(vectors)
    
    return matrix, positions


def extract_specific_layer_keys(tensors, layer_number):
    layer_keys = {}
    for layer in layer_number:
        layer_str = f"layer.{layer}."
        for key, value in tensors.items():
            if layer_str in key:
                layer_keys[key] = value
    return layer_keys


def get_adapter_parameters_dict(path, layer_number=11):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    specific_layer_keys = extract_specific_layer_keys(tensors, layer_number)
    
    return specific_layer_keys


def get_original_shapes(weight_dict):
    return {key: weight_dict[key].shape for key in sorted(weight_dict.keys())}


def restore_from_vector_using_positions(vector, shapes, position):
    matrices = {}
    for key, shape in shapes.items():
        start_pos = position[key]
        size = torch.prod(torch.tensor(shape)).item()
        matrices[key] = vector[start_pos:start_pos+size].view(shape)
    return matrices


def build_para_diff_dataset(base_dir, layer_number=11):
    # base_dir = 'exp_results/bert-base-uncased/rte/rte.lora_r_1.lora_alpha_8.epoch_20.seed_42/'

    matching_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if "last_" in dir_name:
                matching_dirs.append(os.path.join(root, dir_name))

    lora_weight_dicts = {}

    for dir in matching_dirs:
        lora_weight_dict = get_adapter_parameters_dict(os.path.join(dir, "adapter_model.safetensors"), layer_number=layer_number)
        
        save_last_steps = dir.split('_last_')[-1]
        lora_weight_dicts[int(save_last_steps)] = lora_weight_dict


    weight_matrix, positions = flatten_and_merge_weights_and_log_positions(lora_weight_dicts)

    original_shapes = get_original_shapes(lora_weight_dicts[0])


    matching_dirs_dict = {}

    for dir in matching_dirs:
        save_last_steps = os.path.basename(dir).split('_last_')[-1]
        matching_dirs_dict[int(save_last_steps)] = dir

    # Save the matrix and positions (for restoration)

    weight_matrix_dir = os.path.join(base_dir, "weights_matrix.pt")
    positions_dir = os.path.join(base_dir, "weights_positions.pt")
    matching_dirs_dict_dir = os.path.join(base_dir, "checkpoints_dirs.pt")
    original_shapes_dir = os.path.join(base_dir, "original_shapes.pt")

    torch.save(weight_matrix, weight_matrix_dir)
    torch.save(positions, positions_dir)
    torch.save(matching_dirs_dict, matching_dirs_dict_dir)
    torch.save(original_shapes, original_shapes_dir)

    # # Load the model
    weight_matrix = torch.load(weight_matrix_dir)
    positions = torch.load(positions_dir)
    original_shapes = torch.load(original_shapes_dir)

    dataset_path = os.path.join(base_dir, "layer_" + '_'.join([str(i) for i in layer_number]))

    return dataset_path


def build_multi_dataset(base_dir, layer_number=11, rank=4):
    dir_list = os.listdir(base_dir)
    print(f'rank {rank} | build {base_dir}')
    target_dir = None
    for directory in dir_list:
        if f"r_{rank}." in directory:
            target_dir = directory
            break

    # Update base_dir if the target directory is found
    if target_dir:
        base_dir = os.path.join(base_dir, target_dir)
        print(f"rank {rank} base_dir: {base_dir}")

    matching_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if "last_" in dir_name:
                matching_dirs.append(os.path.join(root, dir_name))

    lora_weight_dicts = {}

    for dir in matching_dirs:
        lora_weight_dict = get_adapter_parameters_dict(os.path.join(dir, "adapter_model.safetensors"), layer_number=layer_number)
        
        save_last_steps = dir.split('_last_')[-1]
        lora_weight_dicts[int(save_last_steps)] = lora_weight_dict


    weight_matrix, positions = flatten_and_merge_weights_and_log_positions(lora_weight_dicts)

    original_shapes = get_original_shapes(lora_weight_dicts[0])

    matching_dirs_dict = {}

    for dir in matching_dirs:
        save_last_steps = os.path.basename(dir).split('_last_')[-1]
        matching_dirs_dict[int(save_last_steps)] = dir

    return weight_matrix, positions, original_shapes, matching_dirs_dict


def build_multi_para_diff_dataset(base_dir, layer_number=[11,10], datasets=["rte", "sst2", "cola", "mnli", "qnli"], rank=4):
    dataset_path = os.path.join(base_dir, f'lora_r_{rank}' ,"para_dataset_" + "_".join(datasets), "layer_" + '_'.join([str(i) for i in layer_number]))
    
    if not os.path.exists(os.path.join(dataset_path, "weights_matrix.pt")):
        weight_matrix_dict = {}
        positions_dict = {}
        original_shapes_dict = {}
        matching_dirs_dict = {}

        for dataset in datasets:
            print(f"Processing {dataset} dataset")
            weight_matrix, positions, original_shapes, matching_dirs = build_multi_dataset(os.path.join(base_dir, dataset), layer_number=layer_number, rank=rank)

            weight_matrix_dict[dataset] = weight_matrix
            positions_dict[dataset] = positions
            original_shapes_dict[dataset] = original_shapes
            matching_dirs_dict[dataset] = matching_dirs
            
            print(f"weight_matrix: {weight_matrix.shape} | positions: {len(positions)} | original_shapes: {len(original_shapes)} | matching_dirs: {len(matching_dirs)}")


        os.makedirs(dataset_path, exist_ok=True)
        weight_matrix_dir = os.path.join(dataset_path, "weights_matrix.pt")
        positions_dir = os.path.join(dataset_path, "weights_positions.pt")
        matching_dirs_dict_dir = os.path.join(dataset_path, "checkpoints_dirs.pt")
        original_shapes_dir = os.path.join(dataset_path, "original_shapes.pt")

        torch.save(weight_matrix_dict, weight_matrix_dir)
        torch.save(positions_dict, positions_dir)
        torch.save(matching_dirs_dict, matching_dirs_dict_dir)
        torch.save(original_shapes_dict, original_shapes_dir)
    
    return dataset_path
    


if __name__ == '__main__':
    build_multi_para_diff_dataset(base_dir="exp_results/bert-base-uncased/", 
                                layer_number=[11,10,9,8,7,6,5,4,3,2,1,0], 
                                datasets=['cola', 'sst2', 'rte', 'mnli', 'qnli', 'qqp', 'mrpc', 'stsb'])