import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    import imageio.v3 as iio
except Exception as e:
    iio = None

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
VID_EXTS = (".avi", ".mp4", ".mov", ".mkv")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_jpeg(img_path: Path, out_path: Path):
    im = Image.open(img_path).convert("RGB")
    im.save(out_path, "JPEG", quality=92)


def convert_sequence_folder(src_folder: Path, dst_folder: Path):
    ensure_dir(dst_folder)
    files = sorted([p for p in src_folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    for idx, fp in enumerate(files):
        out_file = dst_folder / f"{idx:03d}.jpg"
        save_jpeg(fp, out_file)


def extract_video_frames(video_path: Path, dst_folder: Path):
    if iio is None:
        raise RuntimeError("imageio.v3 not available. Please install imageio[ffmpeg].")
    ensure_dir(dst_folder)
    idx = 0
    for frame in iio.imiter(video_path):
        # frame is numpy array HxWxC in uint8
        out_file = dst_folder / f"{idx:03d}.jpg"
        Image.fromarray(frame).save(out_file, "JPEG", quality=92)
        idx += 1


def convert_item_to_frames(item: Path, out_dir: Path):
    """Convert either a video file or an image-sequence folder into frames folder."""
    if item.is_file() and item.suffix.lower() in VID_EXTS:
        extract_video_frames(item, out_dir)
    elif item.is_dir():
        # If directory contains images, convert; if contains nested frames already, copy/convert top-level
        img_files = [p for p in item.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if img_files:
            convert_sequence_folder(item, out_dir)
        else:
            # Try one level deeper (some datasets have item/frames/*.jpg)
            subdirs = [d for d in item.iterdir() if d.is_dir()]
            for sd in subdirs:
                img_files = [p for p in sd.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
                if img_files:
                    convert_sequence_folder(sd, out_dir)
                    return
            raise RuntimeError(f"No images found to convert under {item}")
    else:
        raise RuntimeError(f"Unsupported item for conversion: {item}")


def prepare_avenue(base: Path):
    src_train = base / "Datasets" / "Avenue Dataset" / "training_videos"
    src_test = base / "Datasets" / "Avenue Dataset" / "testing_videos"
    dst_train = base / "Datasets" / "avenue" / "training" / "frames"
    dst_test = base / "Datasets" / "avenue" / "testing" / "frames"

    if not src_train.exists() or not src_test.exists():
        print("[Avenue] Source folders not found, skipping.")
        return

    for v in sorted(src_train.glob("*.avi")):
        out_dir = dst_train / v.stem
        print(f"[Avenue][train] {v.name} -> {out_dir}")
        extract_video_frames(v, out_dir)

    for v in sorted(src_test.glob("*.avi")):
        out_dir = dst_test / v.stem
        print(f"[Avenue][test] {v.name} -> {out_dir}")
        extract_video_frames(v, out_dir)


def prepare_ped2(base: Path):
    src_train_root = base / "Datasets" / "UCSD_Anomaly_Dataset" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / "Train"
    src_test_root = base / "Datasets" / "UCSD_Anomaly_Dataset" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / "Test"
    dst_train_root = base / "Datasets" / "ped2" / "training" / "frames"
    dst_test_root = base / "Datasets" / "ped2" / "testing" / "frames"

    if not src_train_root.exists() or not src_test_root.exists():
        print("[Ped2] Source folders not found, skipping.")
        return

    # Train folders (UCSD ped2 train should not include *_gt, but filter defensively)
    for folder in sorted([d for d in src_train_root.iterdir() if d.is_dir()]):
        if folder.name.lower().endswith('_gt'):
            continue
        out_dir = dst_train_root / folder.name
        print(f"[Ped2][train] {folder.name} -> {out_dir}")
        convert_sequence_folder(folder, out_dir)

    # Test folders: skip *_gt ground-truth masks, only convert RGB videos
    for folder in sorted([d for d in src_test_root.iterdir() if d.is_dir()]):
        if folder.name.lower().endswith('_gt'):
            continue
        out_dir = dst_test_root / folder.name
        print(f"[Ped2][test] {folder.name} -> {out_dir}")
        convert_sequence_folder(folder, out_dir)


def prepare_shanghaitech(base: Path):
    """Prepare ShanghaiTech into Datasets/shanghaitech/{training,testing}/frames/"""
    # Potential source roots
    candidates = [
        base / "sanghigh" / "shanghaitech_extracted",
        base / "Datasets" / "shanghaitech_extracted",
        base / "Datasets" / "ShanghaiTech",
    ]
    src_root = next((p for p in candidates if p.exists()), None)
    if src_root is None:
        print("[ShanghaiTech] Source root not found. Please extract the archive to one of:\n"
              f" - {candidates[0]}\n - {candidates[1]}\n - {candidates[2]}")
        return

    # Typical structure: <src_root>/training/ and <src_root>/testing/ with either videos (.avi) or image folders
    splits = {
        "training": (src_root / "training", base / "Datasets" / "shanghaitech" / "training" / "frames"),
        "testing": (src_root / "testing", base / "Datasets" / "shanghaitech" / "testing" / "frames"),
    }

    for split, (src_split, dst_split) in splits.items():
        if not src_split.exists():
            print(f"[ShanghaiTech] {split} source not found at {src_split}, skipping this split.")
            continue

        # Recursively collect media: video files and image-sequence directories
        video_files = [p for p in src_split.rglob('*') if p.is_file() and p.suffix.lower() in VID_EXTS]
        image_dirs = set()
        for img in src_split.rglob('*'):
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                image_dirs.add(img.parent)

        # Process image sequence directories
        for d in sorted(image_dirs):
            out_dir = dst_split / d.stem
            print(f"[ShanghaiTech][{split}] {d} -> {out_dir}")
            convert_item_to_frames(d, out_dir)

        # Process videos
        for v in sorted(video_files):
            out_dir = dst_split / v.stem
            print(f"[ShanghaiTech][{split}] {v} -> {out_dir}")
            convert_item_to_frames(v, out_dir)


def main():
    # Resolve project base assuming this script is at Drone-Guard/tools/
    script_path = Path(__file__).resolve()
    base = script_path.parents[1]
    print(f"Project base: {base}")

    prepare_ped2(base)
    prepare_avenue(base)
    prepare_shanghaitech(base)

    print("Done preparing datasets.")


if __name__ == "__main__":
    main()
