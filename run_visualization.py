"""
Example script for running mechanistic interpretability visualizations
on the cursive transformer model.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from model import get_all_args, get_checkpoint
from data import create_datasets
from sample import generate_helper_fn, GenerationParams

import mech_interp

# Configure matplotlib
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14
})

def load_model(run_id, dataset_name, vocab_size, n_layer, max_seq_length):
    """Helper function to load a model checkpoint"""
    if os.path.exists('best_checkpoint.pt'):
        os.remove('best_checkpoint.pt')
        
    args = get_all_args(False)
    args.wandb_project = 'bigbank_2k'
    args.load_from_run_id = run_id
    args.dataset_name = dataset_name
    args.num_words = 5
    args.max_seq_length = max_seq_length
    args.n_layer = n_layer
    args.vocab_size = vocab_size
    args.block_size = max_seq_length
    
    # Additional parameters from the latest run
    args.downsample_mean = 0.65
    args.downsample_width = 0.1
    args.seed = 1337
    
    torch.manual_seed(args.seed)
    train_dataset, test_dataset = create_datasets(args)
    args.context_block_size = test_dataset.get_text_seq_length()
    args.context_vocab_size = test_dataset.get_char_vocab_size()
    
    model, _, _, _, _ = get_checkpoint(args, sample_only=True)
    return model, test_dataset, args

def main():
    # === CONFIGURATION ===
    text = "the anger of Achilles"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create root output directory
    root_output_dir = "mech_interp_results"
    os.makedirs(root_output_dir, exist_ok=True)
    
    # Create version-specific subdirectory
    run_id = "7coqq2c4"  # Updated to latest checkpoint
    output_dir = os.path.join(root_output_dir, f"{run_id}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different types of visualizations
    attention_dir = os.path.join(output_dir, "attention")
    entropy_dir = os.path.join(output_dir, "attention_entropy")
    token_entropy_dir = os.path.join(output_dir, "token_entropy")
    combined_dir = os.path.join(output_dir, "combined")
    
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(entropy_dir, exist_ok=True)
    os.makedirs(token_entropy_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    # === LOAD LATEST MODEL ===
    print(f"Loading latest model (run_id: {run_id})...")
    model, dataset, args = load_model(
        run_id=run_id,
        dataset_name='bigbank_3500',  # Updated dataset name
        vocab_size=525,
        n_layer=5,
        max_seq_length=1050
    )
    
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    
    # === GENERATE TEXT AND CAPTURE ATTENTION AND LOGITS ===
    print(f"Generating text: '{text}'")
    patterns, cross_patterns, self_hook, cross_hook, logit_hook, logit_outputs_list = mech_interp.setup_attention_hooks(model)
    
    # Register hooks
    hooks = []
    for layer in model.transformer.h:
        hooks.extend([
            layer.attn.register_forward_hook(self_hook),
            layer.cross_attn.register_forward_hook(cross_hook)
        ])
    
    # Add hook for logits
    hooks.append(model.lm_head.register_forward_hook(logit_hook))
    
    # Set up generation parameters
    params = GenerationParams()
    params.temperature = 0.6
    params.do_sample = True
    params.num_steps = args.max_seq_length
    params.n_words = 4  # Ensure this is large enough for our text
    params.verbose = True
    
    # Generate text
    try:
        with torch.no_grad():
            word_list = text.strip().split()
            generated_word_offsets = generate_helper_fn(model, dataset, word_list, params)
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    # Collect logits from hook output
    logits = torch.cat(logit_outputs_list, dim=1)
    print(f"Collected logits shape: {logits.shape}")
    
    # === VISUALIZE ATTENTION FOR EACH LAYER AND HEAD ===
    n_layers = len(model.transformer.h)
    n_heads = model.transformer.h[0].attn.n_head
    
    print(f"Generating visualizations for {n_layers} layers, {n_heads} heads each...")
    
    # Layer-by-layer analysis for first head (as example)
    for layer_idx in range(n_layers):
        # Generate attention plot for first head
        fig = mech_interp.plot_attention_comparison(
            model, dataset, generated_word_offsets, text,
            patterns, cross_patterns,
            layer_idx=layer_idx, head_idx=0,  # First head
            title="Attention Analysis (WORD_TOKEN model)"
        )
        fig.savefig(f"{attention_dir}/attention_layer{layer_idx+1}_head1.png", 
                    bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        # Generate entropy plot for this layer
        fig = mech_interp.plot_entropy_analysis(
            model, dataset, generated_word_offsets, text,
            patterns, cross_patterns, layer_idx=layer_idx
        )
        fig.savefig(f"{entropy_dir}/entropy_layer{layer_idx+1}.png", 
                    bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    # Overall entropy analysis across all layers
    fig = mech_interp.plot_entropy_analysis(
        model, dataset, generated_word_offsets, text,
        patterns, cross_patterns
    )
    fig.savefig(f"{entropy_dir}/entropy_all_layers.png", 
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    # Generate token logit entropy visualization
    fig = mech_interp.plot_token_logit_entropy(
        logits, text, generated_word_offsets
    )
    fig.savefig(f"{token_entropy_dir}/token_logit_entropy.png",
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    # Generate combined entropy analysis
    fig = mech_interp.plot_combined_entropy_analysis(
        model, dataset, generated_word_offsets, text,
        patterns, cross_patterns, logits
    )
    fig.savefig(f"{combined_dir}/combined_entropy_analysis.png",
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"Visualizations saved to directories under {output_dir}/")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds") 