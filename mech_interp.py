"""
Mechanistic interpretability tools for visualizing transformer attention
in the cursive handwriting model.
"""

import os
import numpy as np
import torch
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import time

from model import Transformer
from data import offsets_to_strokes
from sample import word_offsets_to_points, plot_strokes, GenerationParams

# =============== ATTENTION HOOKS ===============

def setup_attention_hooks(model):
    """Set up hooks for capturing attention patterns in the model"""
    self_attn_patterns = {i: {} for i in range(len(model.transformer.h))}
    cross_attn_patterns = {i: {} for i in range(len(model.transformer.h))}
    logit_outputs = []  # To store logits for token entropy calculation
    
    def self_attn_hook(mod, inp, out):
        # Extract query, key, value from the module
        x = inp[0]
        q, k, v = mod.c_attn(x).split(mod.n_embd, dim=2)
        B, T, C = q.size()
        
        # Reshape for multi-head attention
        k = k.view(B, T, mod.n_head, C // mod.n_head).transpose(1, 2)
        q = q.view(B, T, mod.n_head, C // mod.n_head).transpose(1, 2)
        
        # Calculate attention weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mod.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Store attention pattern
        layer_idx = next(i for i, layer in enumerate(model.transformer.h) if layer.attn == mod)
        self_attn_patterns[layer_idx][mod] = att.detach()
        
        return out

    def cross_attn_hook(mod, inp, out):
        # Extract input and context
        x, context = inp
        B, T, C = x.size()
        _, T_ctx, _ = context.size()
        
        # Calculate query, key, value
        q = mod.c_attn_q(x).view(B, T, mod.n_ctx_head, C // mod.n_ctx_head).transpose(1, 2)
        k, v = mod.c_attn_kv(context).split(mod.n_embd_context, dim=2)
        k = k.view(B, T_ctx, mod.n_ctx_head, C // mod.n_ctx_head).transpose(1, 2)
        
        # Calculate attention weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        
        # Store attention pattern
        layer_idx = next(i for i, layer in enumerate(model.transformer.h) if layer.cross_attn == mod)
        cross_attn_patterns[layer_idx][mod] = att.detach()
        
        return out
    
    def logit_hook(mod, inp, out):
        # Store the logits output from the model's head
        logit_outputs.append(out.detach())
        return out
        
    return self_attn_patterns, cross_attn_patterns, self_attn_hook, cross_attn_hook, logit_hook, logit_outputs


# =============== ENTROPY CALCULATION ===============

def calculate_entropy(attention_weights):
    """Calculate entropy of attention distributions"""
    # Ensure we don't have zeros by adding small epsilon
    eps = 1e-10
    attention_weights = attention_weights + eps
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(attention_weights * torch.log2(attention_weights), dim=-1)
    return entropy


def calculate_logit_entropy(logits):
    """Calculate entropy of token distribution from logits"""
    # Apply softmax to get probability distribution
    probs = F.softmax(logits, dim=-1)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    probs = probs + eps
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
    return entropy


# =============== VISUALIZATION FUNCTIONS ===============

def plot_attention_heatmap(attention_matrix, ax, title, cmap="YlOrRd", 
                           xlabel="Position", ylabel="Position", 
                           xticklabels=50, yticklabels=50,
                           vmin=None, vmax=None):
    """Plot a single attention heatmap"""
    import seaborn as sns
    sns.heatmap(
        attention_matrix,
        ax=ax,
        cmap=cmap,
        cbar_kws={'label': f'{title} Weight'},
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        vmin=vmin,
        vmax=vmax,
        rasterized=True
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_attention_comparison(model, dataset, generated_word_offsets, text,
                              patterns, cross_patterns,
                              layer_idx=3, head_idx=3, title="Attention Analysis",
                              figsize=(16, 12), dpi=150):
    """Plot attention patterns for a single model"""
    params = GenerationParams()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.8, 1, 1], hspace=0.3)

    # Convert offsets to points for plotting
    generated_points = word_offsets_to_points(generated_word_offsets, params)
    
    # Plot the handwriting
    ax_writing = fig.add_subplot(gs[0])
    plot_strokes(np.vstack(generated_points), "", fig=fig, ax=ax_writing)
    ax_writing.set_title(f"Generated Writing for: '{text}'", pad=10)
    ax_writing.set_xticks([])
    ax_writing.set_yticks([])

    # Extract attention patterns
    self_attn = next(iter(patterns[layer_idx].values())).squeeze(0).detach().cpu().numpy()
    cross_attn = next(iter(cross_patterns[layer_idx].values())).squeeze(0).detach().cpu().numpy()
    
    # Plot self-attention
    ax_self = fig.add_subplot(gs[1])
    plot_attention_heatmap(
        self_attn[head_idx],
        ax_self,
        title="Self-Attention Pattern",
        xlabel="Generated Sequence Position",
        ylabel="Generated Sequence Position"
    )

    # Plot cross-attention
    ax_cross = fig.add_subplot(gs[2])
    plot_attention_heatmap(
        cross_attn[head_idx],
        ax_cross,
        title="Cross-Attention Pattern",
        xlabel="Input Text Position",
        ylabel="Generated Sequence Position",
        cmap="YlGnBu"
    )

    # Add text labels at the top
    ax_cross_top = ax_cross.twiny()
    ax_cross_top.set_xlim(ax_cross.get_xlim())
    ax_cross_top.set_xticks(np.arange(len(text)))
    ax_cross_top.set_xticklabels(
        list(text),
        rotation=0,
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

    # Add stroke position markers
    seq_len = self_attn[head_idx].shape[0]
    stroke_positions = np.linspace(0, seq_len, min(len(generated_points), seq_len))
    for pos in stroke_positions[::25]:
        ax_self.axvline(x=pos, color='red', alpha=0.05, linestyle='-')
        ax_self.axhline(y=pos, color='red', alpha=0.05, linestyle='-')
        ax_cross.axhline(y=pos, color='red', alpha=0.05, linestyle='-')

    plt.suptitle(
        f"{title}\n" +
        f"Layer {layer_idx + 1}, Head {head_idx + 1}",
        y=0.98,
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_entropy_analysis(model, dataset, generated_word_offsets, text,
                          patterns, cross_patterns, layer_idx=None,
                          figsize=(16, 12), dpi=150):
    """Plot entropy of attention distributions across layers and heads"""
    params = GenerationParams()
    n_layers = len(model.transformer.h)
    n_heads = model.transformer.h[0].attn.n_head
    
    if layer_idx is None:
        # If no specific layer, analyze all layers
        layers_to_analyze = range(n_layers)
    else:
        # Otherwise, just analyze the specified layer
        layers_to_analyze = [layer_idx]
        
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    if len(layers_to_analyze) == 1:
        # Single layer analysis - show entropy per head
        gs = gridspec.GridSpec(2, 2, height_ratios=[0.8, 1], hspace=0.3, wspace=0.3)
        
        # Plot the handwriting
        ax_writing = fig.add_subplot(gs[0, :])
        generated_points = word_offsets_to_points(generated_word_offsets, params)
        plot_strokes(np.vstack(generated_points), "", fig=fig, ax=ax_writing)
        ax_writing.set_title(f"Generated Writing for: '{text}'", pad=10)
        ax_writing.set_xticks([])
        ax_writing.set_yticks([])
        
        # Get the single layer's attention patterns
        layer = layers_to_analyze[0]
        self_attn = next(iter(patterns[layer].values())).squeeze(0).detach().cpu()
        cross_attn = next(iter(cross_patterns[layer].values())).squeeze(0).detach().cpu()
        
        # Calculate entropy
        self_entropy = calculate_entropy(self_attn)  # (n_heads, seq_len)
        cross_entropy = calculate_entropy(cross_attn)  # (n_heads, seq_len)
        
        # Plot self-attention entropy
        ax_self = fig.add_subplot(gs[1, 0])
        for h in range(n_heads):
            ax_self.plot(self_entropy[h].numpy(), label=f'Head {h+1}')
        ax_self.set_title(f"Self-Attention Entropy (Layer {layer+1})")
        ax_self.set_xlabel("Sequence Position")
        ax_self.set_ylabel("Entropy (bits)")
        ax_self.legend()
        ax_self.grid(alpha=0.3)
        
        # Plot cross-attention entropy
        ax_cross = fig.add_subplot(gs[1, 1])
        for h in range(n_heads):
            ax_cross.plot(cross_entropy[h].numpy(), label=f'Head {h+1}')
        ax_cross.set_title(f"Cross-Attention Entropy (Layer {layer+1})")
        ax_cross.set_xlabel("Sequence Position")
        ax_cross.set_ylabel("Entropy (bits)")
        ax_cross.legend()
        ax_cross.grid(alpha=0.3)
        
    else:
        # Multi-layer analysis - show average entropy across layers
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.8, 1, 1], hspace=0.3)
        
        # Plot the handwriting
        ax_writing = fig.add_subplot(gs[0])
        generated_points = word_offsets_to_points(generated_word_offsets, params)
        plot_strokes(np.vstack(generated_points), "", fig=fig, ax=ax_writing)
        ax_writing.set_title(f"Generated Writing for: '{text}'", pad=10)
        ax_writing.set_xticks([])
        ax_writing.set_yticks([])
        
        # Create arrays to store entropy values
        all_self_entropy = []
        all_cross_entropy = []
        
        # Calculate entropy for each layer
        for layer in layers_to_analyze:
            self_attn = next(iter(patterns[layer].values())).squeeze(0).detach().cpu()
            cross_attn = next(iter(cross_patterns[layer].values())).squeeze(0).detach().cpu()
            
            self_entropy = calculate_entropy(self_attn).mean(dim=0).numpy()  # Average across heads
            cross_entropy = calculate_entropy(cross_attn).mean(dim=0).numpy()
            
            all_self_entropy.append(self_entropy)
            all_cross_entropy.append(cross_entropy)
        
        # Plot self-attention entropy
        ax_self = fig.add_subplot(gs[1])
        for i, layer in enumerate(layers_to_analyze):
            ax_self.plot(all_self_entropy[i], label=f'Layer {layer+1}')
        ax_self.set_title("Self-Attention Entropy (Averaged Across Heads)")
        ax_self.set_xlabel("Sequence Position")
        ax_self.set_ylabel("Entropy (bits)")
        ax_self.legend()
        ax_self.grid(alpha=0.3)
        
        # Plot cross-attention entropy
        ax_cross = fig.add_subplot(gs[2])
        for i, layer in enumerate(layers_to_analyze):
            ax_cross.plot(all_cross_entropy[i], label=f'Layer {layer+1}')
        ax_cross.set_title("Cross-Attention Entropy (Averaged Across Heads)")
        ax_cross.set_xlabel("Sequence Position")
        ax_cross.set_ylabel("Entropy (bits)")
        ax_cross.legend()
        ax_cross.grid(alpha=0.3)
    
    plt.suptitle(
        f"Attention Entropy Analysis for '{text}'",
        y=0.98,
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_token_logit_entropy(logits, text, generated_word_offsets, 
                             figsize=(16, 8), dpi=150):
    """Plot entropy of token logits to show model's confidence over time"""
    params = GenerationParams()
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.8, 1], hspace=0.3)
    
    # Plot the handwriting
    ax_writing = fig.add_subplot(gs[0])
    generated_points = word_offsets_to_points(generated_word_offsets, params)
    plot_strokes(np.vstack(generated_points), "", fig=fig, ax=ax_writing)
    ax_writing.set_title(f"Generated Writing for: '{text}'", pad=10)
    ax_writing.set_xticks([])
    ax_writing.set_yticks([])
    
    # Calculate entropy from logits
    entropy = calculate_logit_entropy(logits).cpu().numpy()
    
    # Plot token logit entropy
    ax_entropy = fig.add_subplot(gs[1])
    ax_entropy.plot(entropy, 'r-', linewidth=1.5)
    
    # Add semi-transparent running average
    window_size = min(50, len(entropy) // 5)
    if window_size > 0:
        running_avg = np.convolve(entropy, np.ones(window_size)/window_size, mode='valid')
        padding = np.full((window_size-1)//2, np.nan)
        running_avg_padded = np.concatenate([padding, running_avg, padding])
        ax_entropy.plot(running_avg_padded, 'b-', linewidth=2, alpha=0.7, label='Running Average')
    
    # Add markers for word boundaries if available
    if len(text.split()) > 1:
        word_lengths = [len(word) for word in text.split()]
        cumulative_lengths = np.cumsum([0] + word_lengths[:-1])
        for i, pos in enumerate(cumulative_lengths):
            ax_entropy.axvline(x=pos * len(entropy) / len(text), 
                               color='green', linestyle='--', alpha=0.5,
                               label='Word Boundary' if i == 0 else None)
    
    ax_entropy.set_title("Token Logit Entropy Over Generation Sequence")
    ax_entropy.set_xlabel("Sequence Position")
    ax_entropy.set_ylabel("Entropy (bits)")
    ax_entropy.grid(alpha=0.3)
    if window_size > 0:
        ax_entropy.legend()
    
    plt.suptitle(
        f"Token Logit Entropy Analysis for '{text}'",
        y=0.98,
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_combined_entropy_analysis(model, dataset, generated_word_offsets, text,
                                  patterns, cross_patterns, logits,
                                  figsize=(16, 14), dpi=150):
    """Plot combined analysis of attention entropy and token logit entropy"""
    params = GenerationParams()
    n_layers = len(model.transformer.h)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.8, 1, 1, 1], hspace=0.4)
    
    # Plot the handwriting
    ax_writing = fig.add_subplot(gs[0])
    generated_points = word_offsets_to_points(generated_word_offsets, params)
    plot_strokes(np.vstack(generated_points), "", fig=fig, ax=ax_writing)
    ax_writing.set_title(f"Generated Writing for: '{text}'", pad=10)
    ax_writing.set_xticks([])
    ax_writing.set_yticks([])
    
    # Create arrays to store entropy values from all layers
    all_self_entropy = []
    all_cross_entropy = []
    
    # Calculate attention entropy for each layer
    for layer in range(n_layers):
        self_attn = next(iter(patterns[layer].values())).squeeze(0).detach().cpu()
        cross_attn = next(iter(cross_patterns[layer].values())).squeeze(0).detach().cpu()
        
        self_entropy = calculate_entropy(self_attn).mean(dim=0).numpy()  # Average across heads
        cross_entropy = calculate_entropy(cross_attn).mean(dim=0).numpy()
        
        all_self_entropy.append(self_entropy)
        all_cross_entropy.append(cross_entropy)
    
    # Plot self-attention entropy
    ax_self = fig.add_subplot(gs[1])
    for i in range(n_layers):
        ax_self.plot(all_self_entropy[i], label=f'Layer {i+1}')
    ax_self.set_title("Self-Attention Entropy (Averaged Across Heads)")
    ax_self.set_xlabel("Sequence Position")
    ax_self.set_ylabel("Entropy (bits)")
    ax_self.legend()
    ax_self.grid(alpha=0.3)
    
    # Plot cross-attention entropy
    ax_cross = fig.add_subplot(gs[2])
    for i in range(n_layers):
        ax_cross.plot(all_cross_entropy[i], label=f'Layer {i+1}')
    ax_cross.set_title("Cross-Attention Entropy (Averaged Across Heads)")
    ax_cross.set_xlabel("Sequence Position")
    ax_cross.set_ylabel("Entropy (bits)")
    ax_cross.legend()
    ax_cross.grid(alpha=0.3)
    
    # Calculate and plot token logit entropy
    ax_token = fig.add_subplot(gs[3])
    token_entropy = calculate_logit_entropy(logits).cpu().numpy()
    ax_token.plot(token_entropy, 'r-', linewidth=1.5)
    
    # Add semi-transparent running average for token entropy
    window_size = min(50, len(token_entropy) // 5)
    if window_size > 0:
        running_avg = np.convolve(token_entropy, np.ones(window_size)/window_size, mode='valid')
        padding = np.full((window_size-1)//2, np.nan)
        running_avg_padded = np.concatenate([padding, running_avg, padding])
        ax_token.plot(running_avg_padded, 'b-', linewidth=2, alpha=0.7, label='Running Average')
    
    ax_token.set_title("Token Logit Entropy Over Generation Sequence")
    ax_token.set_xlabel("Sequence Position")
    ax_token.set_ylabel("Entropy (bits)")
    ax_token.grid(alpha=0.3)
    if window_size > 0:
        ax_token.legend()
    
    plt.suptitle(
        f"Combined Entropy Analysis for '{text}'",
        y=0.98,
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig 