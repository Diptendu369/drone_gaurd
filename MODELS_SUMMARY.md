# Models Folder Explanation

## Files and Exports

### `Attention.py`
- **Purpose**: Implements Grouped Query Attention (GQA), a memory-efficient multi-head attention.
- **Key Classes**:
  - `CausalSelfAttention`: Core GQA module.
    - **Inputs**: `x` (B, S, D), optional `mask`, optional `input_pos`.
    - **Outputs**: Attention-applied tensor (B, S, D).
    - **Mechanism**:
      - Projects input to Q, K, V using separate linear layers.
      - Supports more query heads than key/value heads via grouping (`num_heads` vs `num_kv_heads`).
      - Reshapes to (B, S, n_kv, q_per_kv, head_dim) and (B, S, n_kv, 1, head_dim).
      - Applies optional rotary positional embeddings (`pos_embeddings`).
      - Uses `torch.nn.functional.scaled_dot_product_attention` (Flash Attention) for efficiency.
      - Supports KV caching for inference.
- **Exports**: `CausalSelfAttention` class; no standalone functions.

### `Quantizer.py`
- **Purpose**: Vector quantization for discrete latent representations.
- **Key Classes**:
  - `Quantizer`: Wrapper around `vector_quantize_pytorch`.
    - **Inputs**: Embedding tensor (B, C, H, W).
    - **Outputs**: Quantized embedding and codebook usage losses.
    - **Modes**: Supports `VectorQuantize`, `ResidualVQ`, `GroupedResidualVQ`.
    - **Exports**: `Quantizer` class; `__init__`, `forward`, `code_usage`.


### `backbone.py`
- **Purpose**: Provides EfficientX3D backbone for 3D video encoding.
- **Key Functions**:
  - `Efficientnet_X3D()`: Loads pretrained EfficientX3D-XS model.
    - **Inputs**: Video tensor (B, C, T, H, W).
    - **Outputs**: Multi-scale feature maps `s1`, `s2`, `s3`, `s4`.
    - **Mechanism**:
      - Loads local checkpoint first (`../pretrained/efficient_x3d_xs_original_form.pyth`), falls back to URL.
      - Wraps EfficientX3D and adds `CausalSelfAttention` (GQA) after stages.
      - Returns features at 1/8, 1/4, 1/2, and full resolution.
- **Exports**: `Efficientnet_X3D` class; `forward` method.

### `model.py`
- **Purpose**: Main DroneGuard model (encoder + quantizer + decoder).
- **Key Classes**:
  - `DroneGuard`: Full prediction model.
    - **Inputs**: List of frame tensors `[T, C, H, W]`.
    - **Outputs**: Predicted next frame `[1, C, H, W]`.
    - **Components**:
      - Encoder: Processes multi-scale features from backbone via convolutions and GQA.
      - Quantizer: Discretizes encoder embedding.
      - Decoder: Upsamples and concatenates features; reconstructs via transposed convs.
    - **Losses**: L2 (MSE), MSSSIM, gradient, quantizer commitment.
    - **Exports**: `DroneGuard` class; `forward`, `compute_loss`.

### `__init__.py` (if present)
- **Purpose**: Package-level initialization and convenience imports.
- **Typical Exports**:
  - `__version__`
  - Convenience functions like `get_model` (factory for `DroneGuard`).
  - May expose dataset builders or utilities.

## Inter-File Dependencies
- `Attention.py` is imported by `backbone.py` (to add GQA to EfficientX3D) and `model.py` (to use GQA in encoder).
- `Quantizer.py` is imported by `model.py` (to quantize embeddings).
- `backbone.py` is imported by `model.py` (to provide EfficientX3D features).

## Data Flow
1. **Input frames** -> `model.py` (DroneGuard.forward)
2. **DroneGuard** calls `backbone.py` -> EfficientX3D -> multi-scale features
3. **Features** -> `Attention.py` (CausalSelfAttention) -> attended features
4. **Attended features** -> `model.py` encoder convolutions
5. **Encoder output** -> `Quantizer.py` -> discrete latent
6. **Latent** -> `model.py` decoder (upsampling + concatenation)
7. **Decoder output** -> Predicted frame

## Summary
The `models/` folder contains the core attention, quantization, backbone, and full model definitions that enable future-frame prediction with grouped query attention.
