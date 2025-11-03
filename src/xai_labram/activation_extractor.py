import torch
import numpy as np
from einops import rearrange
import json
from dotenv import load_dotenv
import os
import pickle

# Load environment variables (like CIRCLING_DATASET_PATH)
load_dotenv()

# Import the necessary functions and classes from the 'labram_ft' module
# We need:
#   - modeling_finetune: To get the NeuralTransformer class
#   - utils: For helper functions like get_input_chans and load_state_dict
#   - get_models: The factory function from the finetuning script to build the model
from src.labram_ft import modeling_finetune
from src.labram_ft import utils
from src.labram_ft.run_class_finetuning import get_models

class ActivationExtractor:
    """
    A class to load a finetuned LaBraM model and extract activations 
    from its intermediate layers (or "bottlenecks").
    """
    
    def __init__(self, checkpoint_path, device='cpu'):
        """
        Initializes the extractor by loading the model from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the .pth checkpoint file.
            device (str): The device to load the model onto ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        print(f"Loading model from {checkpoint_path} onto {self.device}...")

        # 1. Load the checkpoint file from disk
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 2. Get the model's configuration ('args') saved during training.
        # This is crucial for re-creating the model with the exact same architecture.
        self.model_args = checkpoint['args']
        
        # 3. Re-create the model instance using the factory function from the training script
        self.model = get_models(self.model_args)
        
        # 4. Load the saved model weights (the 'state dictionary')
        # The weights might be nested under 'model' (from a single-GPU save)
        # or 'module' (from a DataParallel save).
        model_state_dict = checkpoint.get('model', checkpoint.get('module', checkpoint))
        utils.load_state_dict(self.model, model_state_dict)
        
        # 5. Move the model to the specified device and set it to evaluation mode
        self.model.to(self.device)
        self.model.eval() # This disables dropout and other training-specific layers
        print("Model loaded successfully.")

        # 6. Define the channel names for the Circling dataset.
        # This exact list comes from 'make_Circling_overlapping.py'.
        self.ch_names = ['FP1','FPZ','FP2',
                         'AF7','AF3','AFZ','AF4','AF8',
                         'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
                         'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
                         'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
                         'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
                         'P9','P7','P5','P3','P1','PZ','P2','P4','P6','P8','P10',
                         'PO7','PO3','POZ','PO4','PO8',
                         'O1','OZ','O2',
                         'IZ'
                        ]
        
        # 7. Get the corresponding channel *indices* from the master list in utils.py.
        # The model's positional embedding is a large tensor, and this tells it 
        # *which* 64 channel embeddings to use for this specific dataset.
        self.input_chans = utils.get_input_chans(self.ch_names)


    @torch.no_grad() # Disable gradient calculations to save memory and compute
    def get_activations(self, eeg_tensor, layer_ids):
        """
        Gets the activations for a single EEG sample from specified layers.

        Args:
            eeg_tensor (torch.Tensor): A raw EEG data tensor. 
                                       Expected shape: (n_channels, n_samples), e.g., (64, 800).
            layer_ids (list of int): A list of layer indices to extract from (e.g., [3, 7, 11]).
        
        Returns:
            dict: A dictionary where keys are layer_ids and values are the 
                  corresponding activation vectors (as 1D numpy arrays).
        """
        
        # --- 1. Preprocessing ---
        
        # The model expects a batch dimension, so add one:
        # (64, 800) -> (1, 64, 800)
        if eeg_tensor.ndim == 2:
            eeg_tensor = eeg_tensor.unsqueeze(0) 
        
        # This preprocessing must be *identical* to the training pipeline:
        # a. Reshape into patches: (1, 64, 800) -> (1, 64, 4, 200)
        #    (B=1, N=64 channels, A=4 time patches, T=200 samples per patch)
        # b. Scale by 100: This is a normalization step used during finetuning.
        eeg_tensor = rearrange(eeg_tensor, 'B N (A T) -> B N A T', T=200)
        eeg_tensor = eeg_tensor.float().to(self.device) / 100

        # --- 2. Activation Extraction ---
        
        # Call the model's 'forward_intermediate' function. We pass:
        #   - eeg_tensor: The preprocessed input
        #   - layer_id: The list of layers we want to get output from
        #   - norm_output=True: Ensures the output is passed through the model's
        #     final normalization layer, just like in a real forward pass.
        #   - input_chans: The list of 64 channel indices we prepared.
        layer_outputs = self.model.forward_intermediate(
            eeg_tensor, 
            layer_id=layer_ids, 
            norm_output=True, 
            input_chans=self.input_chans
        )
        
        # --- 3. Format Output ---
        
        # 'layer_outputs' is a list of tensors, one for each layer_id.
        # Each tensor is the final (B, D) vector for that layer.
        activations = {}
        for i, layer_id in enumerate(layer_ids):
            activation_vector = layer_outputs[i] 
            
            # Squeeze to remove batch dim (1, 200) -> (200,)
            # Move to CPU and convert to a numpy array for use with scikit-learn
            activations[layer_id] = activation_vector.squeeze().cpu().numpy()
            
        return activations

# This block only runs when the script is executed directly
# (e.g., `python -m src.xai_labram.activation_extractor`)
if __name__ == '__main__':
    
    # --- 1. Configuration ---
    CHECKPOINT_PATH = 'models/checkpoints/finetune_circling_v5/checkpoint-best.pth'
    dataset_root = os.getenv("CIRCLING_DATASET_PATH")
    
    # Load the data manifests we created in Step 3
    with open(f'{dataset_root}/tcav_sanity_check_data/concept_open.json', 'r') as f:
        concept_files = json.load(f)
    with open(f'{dataset_root}/tcav_sanity_check_data/random_closed1.json', 'r') as f:
        random_files = json.load(f)
    
    # Get one sample from each class for our validation checks
    SAMPLE_OPEN_PATH = concept_files[0]
    SAMPLE_CLOSED_PATH = random_files[0]
    
    # --- 2. Initialization ---
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ActivationExtractor(CHECKPOINT_PATH, device=device_str)
    
    # Load the raw EEG data for both samples
    with open(SAMPLE_OPEN_PATH, 'rb') as f:
        data = pickle.load(f)
        eeg_data_open = torch.from_numpy(data['X'])
    print(f"\nLoaded 'Open' sample from: {SAMPLE_OPEN_PATH}")
    print(f"Original shape: {eeg_data_open.shape}")

    with open(SAMPLE_CLOSED_PATH, 'rb') as f:
        data = pickle.load(f)
        eeg_data_closed = torch.from_numpy(data['X'])
    print(f"Loaded 'Closed1' sample from: {SAMPLE_CLOSED_PATH}")
    
    # Define the layers we want to inspect
    target_layers = [3, 7, 11] # Early, middle, and late layers
    
    #################################################################
    print("\n--- VALIDATION 1: Replicating Model Prediction ---")
    
    # Goal: Check if the activation from the last layer, when passed
    #       through the classification head, matches a full model forward pass.
    
    # 1. Get the model's "direct" output (the standard way)
    eeg_tensor_batch = eeg_data_open.unsqueeze(0) 
    eeg_tensor_processed = rearrange(eeg_tensor_batch, 'B N (A T) -> B N A T', T=200).float() / 100
    eeg_tensor_processed = eeg_tensor_processed.to(extractor.device)
    
    with torch.no_grad():
        logit_direct = extractor.model(eeg_tensor_processed, input_chans=extractor.input_chans)
    
    # 2. Get the "manual" output using our extractor
    # Get the activation vector from the final block (layer 11)
    activations = extractor.get_activations(eeg_data_open, layer_ids=[11])
    act_layer_11 = activations[11]
    
    # Convert this numpy vector back to a tensor
    act_tensor = torch.from_numpy(act_layer_11).unsqueeze(0).to(extractor.device)
    
    # Pass it through the final classification layer ('head') manually
    with torch.no_grad():
        head = extractor.model.head
        logit_manual = head(act_tensor)
    
    # 3. Compare them. They should be (almost) identical.
    is_close = torch.allclose(logit_direct, logit_manual, atol=1e-5)
    print(f"Direct Logit: {logit_direct.item():.6f}")
    print(f"Manual Logit: {logit_manual.item():.6f}")
    print(f"Match: {is_close}")
    if is_close:
        print("SUCCESS: Logits match. Extractor is verified.")
    else:
        print("ERROR: Logits do not match! The extractor logic is flawed.")

    #################################################################
    print("\n--- VALIDATION 2: Differentiating Classes ---")
    
    # Goal: Check that the model's internal representations for 'Open' and 'Closed1'
    #       are actually different.
    
    # Get activations for both samples from the last layer
    act_open = extractor.get_activations(eeg_data_open, layer_ids=[11])[11]
    act_closed = extractor.get_activations(eeg_data_closed, layer_ids=[11])[11]
    
    # Calculate Cosine Similarity (1.0 = identical, -1.0 = opposite, 0.0 = orthogonal)
    cos_sim = np.dot(act_open, act_closed) / (np.linalg.norm(act_open) * np.linalg.norm(act_closed))
    print(f"Cosine Similarity between 'Open' and 'Closed1' (Layer 11): {cos_sim:.4f}")
    if cos_sim < 0.9:
        print("SUCCESS: Activations are distinct, as expected.")
    else:
        print("Warning: Activations are very similar. Model may not be differentiating well.")

    #################################################################
    print("\n--- VALIDATION 3: Deterministic Output (Stability) ---")
    
    # Goal: Check that running the same input twice produces the exact same output.
    #       This confirms the model is in eval() mode (e.g., dropout is off).
    
    act_v1 = extractor.get_activations(eeg_data_open, layer_ids=target_layers)
    act_v2 = extractor.get_activations(eeg_data_open, layer_ids=target_layers)
    
    is_identical = np.array_equal(act_v1[11], act_v2[11])
    print(f"Activations are identical on two runs: {is_identical}")
    if is_identical:
        print("SUCCESS: Activations are stable and deterministic.")
    else:
        print("ERROR: Activations are not deterministic! Model may be in train() mode.")