# Drone-Guard Project Pipeline and Detailed Explanation

## Project Overview
Drone-Guard is an unsupervised video anomaly detection system that predicts future frames and flags anomalies when reconstruction quality drops. It uses EfficientX3D as a 3D video encoder, Grouped Query Attention (GQA), and vector quantization to learn compact representations. The project supports datasets: Ped2, Avenue, and ShanghaiTech.

---

## Project Pipeline
1. **Dataset Preparation** (`tools/prepare_datasets.py`): Converts raw videos into per-video frame directories.
2. **Data Loading** (`Datasets/video_data.py`): Loads frames, creates temporal windows, and applies transforms.
3. **Model Definition** (`models/model.py`): Defines the DroneGuard model (encoder, quantizer, decoder).
4. **Backbone** (`models/backbone.py`): EfficientX3D provides spatiotemporal features.
5. **Training** (`train.py`): Trains the model with future-frame prediction loss and optional pseudo anomalies.
6. **Testing/Evaluation** (`test.py`): Computes per-frame PSNR, aggregates per-video, and calculates AUC.
7. **Visualization** (`test.py`): Exports ROC curves and per-video PSNR plots to `results/<dataset>/`.

---

## File-by-File Explanation

### `config/`
- `defaults.py`: Base configuration (paths, model defaults).
- `ped2.yaml`, `avenue.yaml`, `shanghaitech.yaml`: Dataset-specific overrides (image size, batch size, learning rate, dataset name, frame steps).

### `tools/prepare_datasets.py`
- **Purpose**: Convert raw datasets into per-video frame folders.
- **Process**:
  - Ped2: Skips `_gt` mask folders, converts image sequences.
  - Avenue: Extracts `.avi` videos to frame directories.
  - ShanghaiTech: Recursively scans for videos or image-sequence folders.
- **Output**: `Datasets/<dataset>/training/frames/<video>/` and `.../testing/frames/<video>/`.

### `Datasets/video_data.py`
- **Purpose**: PyTorch Dataset for loading frames and creating temporal windows.
- **Key Steps**:
  - Walks each video folder, sorts frames.
  - For training: creates overlapping windows of `NUM_FRAMES` with jumps.
  - For testing: creates sliding windows for inference.
  - Applies transforms (resize, normalize, tensor).
- **Outputs**: Tensors of shape `[T, C, H, W]` (temporal sequence).

### `models/backbone.py`
- **Purpose**: 3D spatiotemporal encoder using EfficientX3D.
- **Components**:
  - Loads pretrained EfficientX3D (XS expansion) from local or URL.
  - Returns multi-scale features (`s1`, `s2`, `s3`, `s4`).
  - Uses Grouped Query Attention (GQA) to refine features.
- **Outputs**: Feature maps at different spatial resolutions.

### `models/model.py`
- **Purpose**: Full DroneGuard model (encoder + quantizer + decoder).
- **Encoder**:
  - Processes multi-scale features from backbone.
  - Applies convolutions and GQA.
  - Produces an embedding for quantization.
- **Quantizer** (`models/Quantizer.py`):
  - VectorQuantize learns a discrete codebook.
  - Commitment loss encourages codebook usage.
- **Decoder**:
  - Upsamples and concatenates features.
  - Uses transposed convolutions and conv layers.
  - Outputs the predicted future frame.
- **Losses**:
  - L2 (MSE) between predicted and ground-truth future frame.
  - MSSSIM (multi-scale SSIM) for perceptual quality.
  - Gradient loss (optional) for edge sharpness.
  - Commitment loss from quantizer.

### `train.py`
- **Purpose**: Train the model with future-frame prediction.
- **Loop**:
  - Loads training and test datasets.
  - For each batch: runs model forward, computes losses.
  - Optimizes with Lion or Adam.
  - Periodically evaluates on test set (computes PSNR, AUC).
  - Saves checkpoints (`epoch_*.pth`, `final_state.pth`).

### `test.py`
- **Purpose**: Evaluate a trained model and export results.
- **Process**:
  - Loads model checkpoint.
  - Runs inference on test videos frame-by-frame.
  - Computes PSNR for each predicted frame.
  - Aggregates per-video PSNR lists.
  - Calculates AUC against ground-truth labels.
  - Exports ROC curve, per-video PSNR plots, and CSV.

### `utils/`
- `train_util.py`: Batching, frame conversion utilities.
- `log_util.py`: Logging and TensorBoard setup.
- `loss_util.py`: Loss functions (L2, MSSSIM, gradient).
- `anomaly_util.py`: PSNR calculation, AUC computation, label loading.

---

## Answers to Your Questions

### 1. What is the frame rate used in the project?
- The project does not enforce a specific frame rate. It uses the original frame sequence as provided by the dataset. Temporal windows are created based on `NUM_FRAMES` and `FRAME_STEPS` in the config, not on time.

### 2. How is the video broken down into frames?
- Raw videos are converted to individual JPEG frames by `tools/prepare_datasets.py`. Each video becomes a folder of sequentially numbered frames (`001.jpg`, `002.jpg`, ...). The dataset loader then creates sliding windows of `NUM_FRAMES` consecutive frames.

### 3. What preprocessing is done?
- Resize to `MODEL.IMAGE_SIZE` (e.g., 192x288 or 224x288) and power-of-2 padding for EfficientX3D.
- Normalize to [-1, 1] (mean=0.5, std=0.5 per channel).
- Convert to PyTorch tensors.

### 4. What is the need of 3D video? What does EfficientX3D take as input and return?
- 3D video processing captures spatiotemporal features (motion and appearance) crucial for anomaly detection.
- **Input to EfficientX3D**: A tensor of shape `[B, C, T, H, W]` (batch, channels, temporal, height, width). In this project, it receives a short clip (e.g., 4 frames) at a time.
- **Output**: Multi-scale feature maps (`s1`, `s2`, `s3`, `s4`) at different spatial resolutions, each with temporal dimension preserved.

### 5. Are transformers used in EfficientX3D? How do they work?
- EfficientX3D itself is a convolutional 3D network, not a transformer. However, this project adds a **Grouped Query Attention (GQA)** module after the backbone to refine features. GQA is a transformer-style attention mechanism that groups queries and shares keys/values across groups, reducing compute while preserving performance.

### 6. What is the use of the attention mechanism and how does it work?
- **Purpose**: To focus on important regions and features across space and time, improving representation quality.
- **Mechanism (GQA)**:
  - Input: Feature map from the backbone.
  - Computes queries, keys, values via linear projections.
  - Groups queries; each group attends to shared keys/values.
  - Produces attention-weighted feature maps.
  - Helps the model prioritize moving objects or salient areas.

### 7. What other innovations are involved?
- **Vector Quantization**: Learns a discrete codebook for latent representations, encouraging efficient encoding.
- **Future-Frame Prediction**: The model is trained to predict the next frame; anomalies are detected when prediction error is high.
- **Multi-Scale Features**: Uses features from multiple encoder stages for better reconstruction.

### 8. How to identify background vs. object, static vs. dynamic features?
- The model learns implicitly: static backgrounds are predictable and yield low reconstruction error; dynamic objects (especially anomalies) cause higher error. Attention mechanisms help focus on moving regions, but there is no explicit segmentation.

### 9. How is error calculated in frames?
- **MSE (L2 Loss)**: Pixel-wise mean squared error between predicted and ground-truth frames.
- **PSNR**: Computed from MSE: `PSNR = 10 * log10(MAX_I^2 / MSE)`, where `MAX_I=1.0` for normalized images. Higher PSNR means better reconstruction.
- **MSSSIM**: Multi-scale structural similarity for perceptual quality.
- **Gradient Loss**: Encourages edge similarity.

### 10. What is the predefined threshold in this project?
- There is no fixed threshold. During evaluation, per-frame PSNR values are aggregated per video, and AUC is computed over varying thresholds. Practitioners can choose a threshold based on desired false positive rate.

### 11. Did we use MSE? Did we use PSNR? Explain PSNR in detail.
- **MSE**: Yes, as part of the total loss (L2 loss).
- **PSNR**: Yes, for evaluation. PSNR (Peak Signal-to-Noise Ratio) measures reconstruction quality:
  - Compute MSE between predicted and ground-truth frames (after denormalizing to [0,1]).
  - PSNR = 10 * log10(MAX / sqrt(MSE)) (since max pixel value is 1).
  - Higher PSNR indicates lower error; anomalies typically show lower PSNR.

### 12. Why PSNR? Why PSNR over other error calculations?
- PSNR is standard in video quality and anomaly detection literature, providing a single scalar per frame that correlates with perceptual quality. It’s easy to compute and compare across methods. While other metrics (SSIM, LPIPS) exist, PSNR is sufficient for ranking anomalies.

### 13. How are anomalies linked with the attention mechanism?
- Attention helps the model focus on salient regions, improving reconstruction of normal patterns. When an anomaly occurs, attention may not adequately explain it, leading to higher reconstruction error (lower PSNR). Thus, attention indirectly influences anomaly scoring.

### 14. Detailed explanation of attention mechanism usage
- After EfficientX3D, features pass through GQA:
  - Features are reshaped to (N, C, H*W) to treat spatial locations as tokens.
  - Linear layers project to Q, K, V.
  - Q is grouped; each group attends to shared K, V.
  - Output is reshaped back to feature maps.
- This allows the model to weight important regions (e.g., moving objects) more heavily, improving prediction accuracy for normal scenes.

### 15. How are we focusing on important regions? Describe the mechanism.
- GQA assigns higher attention weights to regions with informative features (e.g., motion, edges). During future-frame prediction, these regions are reconstructed more accurately. Anomalous regions, which are out-of-distribution, receive less appropriate attention, resulting in higher error.

### 16. What is the use of Grad-CAM and its contribution?
- Grad-CAM is not currently used in this project. It could be added to visualize which regions the model attends to during prediction, helping interpretability.

### 17. Where is the proper training code?
- `train.py` contains the main training loop. It loads datasets, initializes the model, optimizer, and runs epochs with periodic evaluation.

### 18. Where are we importing the model?
- In `train.py` and `test.py`:
  ```python
  from models.model import DroneGuard as get_model
  model = get_model(config)
  ```

### 19. Where is the model explained? Input to EfficientX3D to fully understand the code
- `models/model.py` defines the full DroneGuard model.
- `models/backbone.py` defines the EfficientX3D backbone and GQA.
- Input flow:
  1. A sequence of frames (e.g., 4 frames) is passed to EfficientX3D.
  2. EfficientX3D outputs multi-scale features (`s1`, `s2`, `s3`, `s4`).
  3. These features are processed by convolutions and GQA in `DroneGuard`.
  4. The embedding is quantized.
  5. The decoder reconstructs the next frame.

### 20. How is the anomaly defined?
- Anomaly is defined as a significant drop in reconstruction quality (low PSNR) compared to normal patterns. During testing, per-frame PSNR is computed; frames below a chosen threshold are flagged as anomalous. The AUC metric evaluates detection quality across all thresholds.

---

## Detailed Execution Flow (Code Files in Order)

### 1. Dataset Preparation
- **File**: `tools/prepare_datasets.py`
- **Execution**: `python tools/prepare_datasets.py`
- **Process**:
  - Scans raw dataset locations.
  - For Ped2: Converts image sequences to JPEG frames, skips `_gt` folders.
  - For Avenue: Extracts `.avi` videos to frame directories.
  - For ShanghaiTech: Recursively scans for videos or image-sequence folders.
- **Returns**: Frame directories under `Datasets/<dataset>/training/frames/<video>/` and `Datasets/<dataset>/testing/frames/<video>/`.
- **Key Functions**:
  - `prepare_ped2(base)`: Returns nothing; writes frames to disk.
  - `prepare_avenue(base)`: Returns nothing; writes frames to disk.
  - `prepare_shanghaitech(base)`: Returns nothing; writes frames to disk.

### 2. Training
- **File**: `train.py`
- **Execution**: `python train.py --cfg config/<dataset>.yaml --test True`
- **Process**:
  - Loads config (`config/defaults.py` + dataset-specific YAML).
  - Initializes model (`models/model.py` -> `DroneGuard`).
  - Loads backbone (`models/backbone.py` -> `Efficientnet_X3D`).
  - Loads training and test datasets (`Datasets/video_data.py` -> `Video`).
  - For each epoch:
    - Trains on batches; computes losses (L2, MSSSIM, gradient, quantizer commitment).
    - Periodically evaluates on test set (computes PSNR per frame, aggregates per video, calculates AUC).
    - Saves checkpoints (`epoch_*.pth`, `final_state.pth`) to `output/<dataset>/<cfg_name>/`.
    - Logs to `log/<dataset>/Drone-Guard/...` and TensorBoard.
- **Returns**: Trained model checkpoint files and logs.
- **Key Functions**:
  - `train(config)`: Main training loop; returns nothing.
  - `validate(config, model, test_loader)`: Computes PSNR and AUC; returns AUC value.

### 3. Testing/Evaluation
- **File**: `test.py`
- **Execution**: `python test.py --cfg config/<dataset>.yaml --model-file output/<dataset>/<cfg>/final_state.pth`
- **Process**:
  - Loads config and model checkpoint.
  - Loads test dataset (`Datasets/video_data.py`).
  - Runs inference frame-by-frame:
    - For each window, predicts next frame.
    - Computes PSNR between predicted and ground-truth.
    - Aggregates per-video PSNR lists.
  - Calculates AUC against ground-truth labels (`utils/anomaly_util.py`).
  - Exports:
    - ROC curve to `results/<dataset>/roc_curve.png`.
    - Per-video PSNR plots to `results/<dataset>/psnr_video_XXX.png`.
    - PSNR data to `results/<dataset>/psnr_list.csv` and `.npy`.
- **Returns**: AUC score and visual results in `results/<dataset>/`.
- **Key Functions**:
  - `inference(config, model, test_loader)`: Computes PSNR lists; returns `psnr_list`, `labels`.
  - `anomaly_util.calculate_auc(...)`: Computes AUC; returns `auc, fpr, tpr`.

### 4. Visualization 
- **Files**: `visualize_demo.py`, `grad_cam_demo.py`
- **Execution**:
  - `python visualize_demo.py --cfg config/<dataset>.yaml --model-file ...`
  - `python grad_cam_demo.py --cfg config/<dataset>.yaml --model-file ... --target-layer conv_x8.conv.0`
- **Process**:
  - Loads model and test data.
  - Samples videos and frames.
  - `visualize_demo.py`: Saves side-by-side Ground Truth | Predicted | Error maps.
  - `grad_cam_demo.py`: Saves Ground Truth | Predicted | Error | Grad-CAM heatmaps.
- **Returns**: PNG images in `demo_visuals/` or `gradcam_visuals/`.
- **Key Functions**:
  - `save_frame_comparison(...)`: Writes concatenated PNG.
  - `save_frame_with_gradcam(...)`: Writes 4-panel PNG with CAM overlay.

### 5. Utility Functions
- **`utils/train_util.py`**: Batching helpers (`decode_input`, `To_Batch`, `To_Frame`).
- **`utils/log_util.py`**: Logger setup; creates output directories.
- **`utils/loss_util.py`**: Loss functions (`l2_loss`, `msssim`, `gradient_loss`).
- **`utils/anomaly_util.py`**: PSNR calculation (`psnr_park`), AUC computation (`calculate_auc`), label loading.

### 6. Configuration
- **`config/defaults.py`**: Base config class (`_C`); defines defaults.
- **`config/<dataset>.yaml`**: Overrides for dataset-specific parameters (image size, batch size, learning rate, dataset name, frame steps).
- **`update_config(cfg, args)`**: Merges CLI args into config; returns updated cfg.

### 7. Model Components
- **`models/model.py`**:
  - `DroneGuard(config)`: Initializes encoder, quantizer, decoder.
  - `forward(x)`: Returns predicted future frame.
  - `compute_loss(...)`: Returns total loss and components.
- **`models/backbone.py`**:
  - `Efficientnet_X3D()`: Loads pretrained EfficientX3D.
  - `forward(x)`: Returns multi-scale features (`s1`, `s2`, `s3`, `s4`).
- **`models/Quantizer.py`**:
  - `Quantizer(...)`: Wraps VectorQuantize/ResidualVQ.
  - `forward(x)`: Returns quantized embedding and loss.

## Summary
Drone-Guard uses 3D video encoding (EfficientX3D) and attention (GQA) to predict future frames. Anomalies are detected when prediction error (low PSNR) is high. The pipeline includes dataset preparation, training with future-frame prediction, and evaluation with PSNR/AUC. The project supports multiple datasets and exports visual results for analysis.
