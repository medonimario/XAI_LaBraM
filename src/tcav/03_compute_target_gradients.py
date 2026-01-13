import torch
import numpy as np
import os
import pickle
import json
import argparse
from tqdm import tqdm
from einops import rearrange
from dotenv import load_dotenv

from src.xai_labram.activation_extractor import ActivationExtractor

def get_averaged_gradient(extractor, eeg_tensor_raw, layer_id, target_class_idx=None):
    """
    Computes the gradient of the target class logit w.r.t the output of a specific layer.
    
    Args:
        extractor: Instance of ActivationExtractor containing the model.
        eeg_tensor_raw: Input EEG tensor (raw).
        layer_id: Integer index of the layer to hook.
        target_class_idx: The index of the output class to explain. 
                          If None, defaults to the argmax of the output (predicted class).
    
    Returns:
        Numpy array of the pooled gradient.
    """
    model = extractor.model
    model.eval()
    model.zero_grad()
    
    gradient_value = None
    hook_handle_bwd = None

    # --- 1. Define Backward Hook ---
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradient_value
        # grad_output is a tuple; usually the first element is the gradient tensor
        if grad_output[0] is not None:
            gradient_value = grad_output[0].detach().clone()

    # --- 2. Register Hook ---
    if layer_id < 0 or layer_id >= len(model.blocks):
        print(f"Error: Layer {layer_id} out of bounds.")
        return None
        
    target_module = model.blocks[layer_id]
    hook_handle_bwd = target_module.register_full_backward_hook(backward_hook)

    # --- 3. Preprocessing (Preserved from your snippet) ---
    # Ensure 4D shape: [Batch, Channel, Height, Width] logic for the transformer
    if eeg_tensor_raw.ndim == 2: 
        eeg_tensor = eeg_tensor_raw.unsqueeze(0)
    else: 
        eeg_tensor = eeg_tensor_raw
        
    # Reshape for LaBraM: B N (A T) -> B N A T
    eeg_tensor = rearrange(eeg_tensor, 'B N (A T) -> B N A T', T=200)
    eeg_tensor = eeg_tensor.float().to(extractor.device) / 100

    try:
        # --- 4. Forward Pass ---
        # We need the logits to backpropagate from
        logits = model(eeg_tensor, input_chans=extractor.input_chans)
        
        # --- 5. Backward Pass ---
        # We must select a scalar value to backpropagate from.
        # TCAV requires the gradient of the *target class* logit.
        if target_class_idx is not None:
            # Backprop from the specific target class
            score = logits[0, target_class_idx]
        else:
            # If no class specified, backprop from the predicted class (max logit)
            # OR if the model outputs a single scalar (binary), use that.
            if logits.numel() == 1:
                score = logits
            else:
                score = logits.max()
        
        score.backward()
        
    except Exception as e:
        print(f"Error during forward/backward pass at layer {layer_id}: {e}")
        return None
    finally:
        # Always remove hooks to prevent memory leaks or side effects
        if hook_handle_bwd: 
            hook_handle_bwd.remove()

    if gradient_value is None: 
        return None

    # --- 6. Pooling ---
    # Average over the sequence/token dimension (dim 1) to get a single vector per channel/sample
    if gradient_value.shape[1] > 1:
        gradient_pooled = gradient_value[:, 1:, :].mean(dim=1)
    else:
         gradient_pooled = gradient_value.mean(dim=1)

    gradient_np = gradient_pooled.squeeze().cpu().numpy()
    
    # Check for validity
    if np.isnan(gradient_np).any(): 
        return None
        
    return gradient_np


def main(args):
    load_dotenv()
    
    # 1. Setup Directories
    output_dir = os.path.join(args.output_dir, "target_gradients")
    os.makedirs(output_dir, exist_ok=True)
    
    target_layers = [int(l) for l in args.target_layers.split(',')]
    
    # 2. Load Model
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {args.checkpoint_path}...")
    extractor = ActivationExtractor(args.checkpoint_path, device=device_str)

    # 3. Load Target Data
    # We look for the manifest file (e.g., target_class_set.json)
    manifest_path = os.path.join(args.manifest_dir, f"{args.target_class_name}.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
    with open(manifest_path, 'r') as f: 
        target_files = json.load(f)
    
    print(f"Found {len(target_files)} target files. Loading max {args.max_samples}...")
    
    # Load the RAW tensors
    target_eeg_raw = []
    loaded_count = 0
    for f_path in tqdm(target_files, desc="Loading Data"):
        if loaded_count >= args.max_samples: break
        try:
            with open(f_path, 'rb') as f: 
                data = pickle.load(f)
                target_eeg_raw.append(torch.from_numpy(data['X']))
                loaded_count += 1
        except Exception as e:
            print(f"Skipping broken file {f_path}: {e}")

    # 4. Compute Gradients per Layer
    print(f"Computing gradients for layers: {target_layers}")
    
    for layer_id in target_layers:
        layer_output_file = os.path.join(output_dir, f"target_gradients_layer_{layer_id}.pkl")
        
        # Optional: Skip if already exists
        if os.path.exists(layer_output_file) and not args.overwrite:
            print(f"Layer {layer_id} gradients already exist. Skipping...")
            continue

        grads_list = []
        for i in tqdm(range(len(target_eeg_raw)), desc=f"Layer {layer_id}"):
            eeg_tensor = target_eeg_raw[i]
            
            grad = get_averaged_gradient(
                extractor, 
                eeg_tensor, 
                layer_id, 
                target_class_idx=args.target_class_idx
            )
            
            if grad is not None:
                grads_list.append(grad)
        
        # Save to disk
        grads_arr = np.array(grads_list)
        print(f"  Saving {grads_arr.shape[0]} gradients for Layer {layer_id}...")
        
        with open(layer_output_file, 'wb') as f:
            pickle.dump(grads_arr, f)

    print("\nGradient computation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Target Gradients for TCAV")
    
    # Input/Output paths
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--manifest_dir", type=str, required=True, help="Dir containing target_class_set.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save gradients")
    
    # Configuration
    parser.add_argument("--target_class_name", type=str, default="target_class_set", help="Name of JSON file without extension")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", help="Comma separated layer IDs")
    parser.add_argument("--max_samples", type=int, default=100, help="Max number of target samples to process")
    
    # Model Specifics
    parser.add_argument("--target_class_idx", type=int, default=None, 
                        help="The index of the output neuron to explain. Default: None (uses argmax or scalar).")
    
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing gradient files")

    args = parser.parse_args()
    main(args)