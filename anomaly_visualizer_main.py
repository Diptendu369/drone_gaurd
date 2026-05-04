import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2

import Datasets
from utils import train_util, anomaly_util
from config.defaults import _C as config, update_config
from models.model import DroneGuard as get_model

def denorm(tensor):
    """Denormalize from [-1,1] to [0,1]"""
    return (tensor + 1.0) / 2.0

def compute_error_map(gt, pred):
    """Compute pixel-wise absolute error map"""
    err = torch.abs(gt - pred).mean(dim=0)  # [H, W]
    return err

def detect_anomaly_regions(error_map, threshold=0.3, min_area=100):
    """
    Detect anomaly regions using connected components on error map.
    Returns list of bounding boxes [(x, y, w, h), ...]
    """
    # Convert to numpy
    err_np = error_map.cpu().numpy()
    
    # Normalize to 0-255
    err_norm = (err_np / err_np.max() * 255).astype(np.uint8) if err_np.max() > 0 else err_np.astype(np.uint8)
    
    # Threshold to get high-error regions
    thresh_val = int(threshold * 255)
    _, thresh = cv2.threshold(err_norm, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    # Filter by minimum area (exclude background which is label 0)
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            bboxes.append((x, y, w, h))
    
    return bboxes

def draw_bboxes_on_image(img, bboxes, color='red', linewidth=2):
    """
    Draw bounding boxes on image using OpenCV.
    img: numpy array (H, W, C) in [0, 1]
    Returns: numpy array with bboxes in [0, 1]
    """
    # Ensure img is numpy array and C-contiguous
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    # Ensure contiguous memory layout and correct dtype
    img = np.ascontiguousarray(img)
    
    # Convert to uint8 for OpenCV (RGB format)
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Ensure 3 channels
    if len(img_uint8.shape) == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    elif img_uint8.shape[2] == 1:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    
    # Color mapping (BGR for OpenCV)
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255)
    }
    cv_color = color_map.get(color, (0, 0, 255))
    
    # Draw rectangles
    for (x, y, w, h) in bboxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img_uint8, (x, y), (x + w, y + h), cv_color, linewidth)
    
    # Normalize back to [0, 1]
    return img_uint8.astype(np.float32) / 255.0

def save_two_panel_visual(normal_img, abnormal_img, idx, out_dir, video_name, is_anomaly):
    """
    Save 2-panel visualization: Normal Frame | Abnormal Frame
    """
    # Convert tensors to numpy if needed
    if isinstance(normal_img, torch.Tensor):
        normal_img = normal_img.permute(1, 2, 0).cpu().numpy()
    if isinstance(abnormal_img, torch.Tensor):
        abnormal_img = abnormal_img.permute(1, 2, 0).cpu().numpy()
    
    # Ensure same height
    h1, w1 = normal_img.shape[:2]
    h2, w2 = abnormal_img.shape[:2]
    
    if h1 != h2:
        # Resize abnormal to match normal
        abnormal_img = cv2.resize(abnormal_img, (w1, h1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Normal frame
    axes[0].imshow(normal_img)
    axes[0].set_title('Normal Frame', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Abnormal frame
    axes[1].imshow(abnormal_img)
    if is_anomaly:
        axes[1].set_title('Abnormal Frame', fontsize=14, fontweight='bold', color='red')
    else:
        axes[1].set_title('Normal Frame', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Sanitize video name
    safe_name = video_name.replace('\\', '_').replace('/', '_')
    
    # Create video-specific folder
    video_out_dir = os.path.join(out_dir, safe_name)
    os.makedirs(video_out_dir, exist_ok=True)
    
    filename = f"frame_{idx:04d}.png"
    filepath = os.path.join(video_out_dir, filename)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Generate anomaly visualizations with bounding boxes')
    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/ped2.yaml', type=str)
    parser.add_argument('--model-file', help='model checkpoint',
                        default='output/ped2/ped2/final_state.pth', type=str)
    parser.add_argument('--num-videos', type=int, default=12, help='Number of videos to process')
    parser.add_argument('--threshold', type=float, default=0.3, help='Error threshold for anomaly detection')
    parser.add_argument('--min-area', type=int, default=100, help='Minimum area for anomaly region')
    parser.add_argument('--out-dir', type=str, default='yolo_visuals', help='Output folder')
    parser.add_argument('opts', help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    update_config(config, args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpus = [config.GPUS[0]] if torch.cuda.is_available() else []

    model = get_model(config)
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])

    # Load checkpoint
    state_dict = torch.load(args.model_file, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if isinstance(model, nn.DataParallel):
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_dataset = eval('Datasets.get_test_data')(config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    os.makedirs(args.out_dir, exist_ok=True)

    loss_func_mse = nn.MSELoss(reduction='none')
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df

    print(f"Processing videos for anomaly detection (threshold={args.threshold})...")
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="Processing videos")):
            if i >= args.num_videos:
                break
            
            video, video_name = train_util.decode_input(input=data, train=False)
            video = [frame.to(device=device) for frame in video]
            video_name = str(video_name[0])

            total_frames = len(video) - fp
            if total_frames <= 0:
                continue

            # Process all frames in video
            for f in range(0, total_frames):
                inputs = video[f:f + fp]
                target = video[f + fp:f + fp + 1][0]

                # Forward pass
                output = model(inputs)

                # Denorm for visualization
                gt_vis = denorm(target[0].cpu())
                pred_vis = denorm(output[0].cpu())

                # Compute error map
                error_map = compute_error_map(gt_vis, pred_vis)
                
                # Detect anomaly regions
                bboxes = detect_anomaly_regions(error_map, threshold=args.threshold, min_area=args.min_area)
                
                # Determine if anomaly present
                is_anomaly = len(bboxes) > 0
                
                # Prepare images
                gt_np = gt_vis.permute(1, 2, 0).cpu().numpy()
                
                if is_anomaly:
                    # Draw bounding boxes on GT image
                    abnormal_img = draw_bboxes_on_image(gt_np, bboxes, color='red', linewidth=2)
                else:
                    # No anomaly: use same image
                    abnormal_img = gt_np.copy()
                
                # Save 2-panel visualization
                save_two_panel_visual(gt_np, abnormal_img, f, args.out_dir, video_name, is_anomaly)

    print(f"Anomaly visualizations saved to: {args.out_dir}")
    print(f"Each video has its own folder with frame-by-frame comparisons.")

if __name__ == '__main__':
    main()  # noqa: E501
