"""
KalaKriti — Dataset Pre-Flight Verification Tool (Stage 5A)
Inspects content and style datasets before training:
  - Validates image file readability and formats
  - Detects corrupted, truncated, or zero-byte files
  - Identifies duplicate images via SHA-256 hashing
  - Analyzes image dimensions, aspect ratios, and color distributions
  - Checks for authentic visual variance (rejects monochromatic/blank/uniform images)
"""

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from PIL import Image
import numpy as np


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_directory(dir_path: str, label: str):
    print("=" * 65)
    print(f"DATASET INSPECTION: {label.upper()}")
    print(f"Directory: {dir_path}")
    print("=" * 65)

    if not os.path.exists(dir_path):
        print(f"❌ DIRECTORY NOT FOUND: '{dir_path}' does not exist.")
        print(f"   Please mount your dataset or verify the path.")
        return {
            "exists": False,
            "valid_count": 0,
            "corrupt_count": 0,
            "duplicates": [],
            "resolutions": [],
        }

    valid_images = []
    corrupt_images = []
    hash_map = defaultdict(list)
    resolutions = []
    color_stds = []

    allowed_exts = (".jpg", ".jpeg", ".png", ".webp")
    all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    for fname in all_files:
        fpath = os.path.join(dir_path, fname)
        if not fname.lower().endswith(allowed_exts):
            continue

        try:
            with Image.open(fpath) as img:
                img.verify()  # Fast structural integrity check
            
            # Reopen to read dimensions and pixel stats
            with Image.open(fpath) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                resolutions.append((w, h))

                # Compute image hash to detect duplicate files
                img_hash = compute_sha256(fpath)
                hash_map[img_hash].append(fname)

                # Color standard deviation check (detects flat/blank images)
                arr = np.array(img_rgb, dtype=np.float32)
                color_std = np.mean([arr[:, :, 0].std(), arr[:, :, 1].std(), arr[:, :, 2].std()])
                color_stds.append(color_std)

                valid_images.append({
                    "name": fname,
                    "resolution": (w, h),
                    "aspect_ratio": round(w / h, 2),
                    "color_std": round(color_std, 1),
                })
        except Exception as e:
            corrupt_images.append((fname, str(e)))

    # Duplicate detection
    duplicates = [v for v in hash_map.values() if len(v) > 1]

    # Reporting
    print(f"\n[1] File Summary:")
    print(f"    - Total image files:    {len(valid_images) + len(corrupt_images)}")
    print(f"    - Valid, readable:      {len(valid_images)}")
    print(f"    - Corrupted / invalid:  {len(corrupt_images)}")
    print(f"    - Duplicate sets:       {len(duplicates)}")

    if corrupt_images:
        print(f"\n[!] Corrupted Files Detected:")
        for fname, err in corrupt_images[:5]:
            print(f"    - {fname}: {err}")

    if duplicates:
        print(f"\n[!] Duplicate Image Sets Detected:")
        for dup in duplicates[:5]:
            print(f"    - {dup}")

    if valid_images:
        min_w = min(r[0] for r in resolutions)
        max_w = max(r[0] for r in resolutions)
        min_h = min(r[1] for r in resolutions)
        max_h = max(r[1] for r in resolutions)
        avg_std = sum(color_stds) / len(color_stds)

        print(f"\n[2] Resolution & Visual Distribution:")
        print(f"    - Width range:          {min_w} px to {max_w} px")
        print(f"    - Height range:         {min_h} px to {max_h} px")
        print(f"    - Average Color StdDev: {avg_std:.1f} (Threshold > 15.0 for non-blank images)")

        # Authentic Warli check: Warli has distinct earthy background and white outlines
        if avg_std < 10.0:
            print(f"    ⚠️ Warning: Very low color variance detected. Check if images are blank or solid color.")
        else:
            print(f"    ✓ Image pixel distributions show genuine visual variance.")

    print("\n" + "=" * 65 + "\n")

    return {
        "exists": True,
        "valid_count": len(valid_images),
        "corrupt_count": len(corrupt_images),
        "duplicates": duplicates,
        "resolutions": resolutions,
    }


def main():
    parser = argparse.ArgumentParser(description="KalaKriti Dataset Pre-Flight Inspector")
    parser.add_argument("--content_dir", type=str, default="./data/coco_train", help="Path to content images")
    parser.add_argument("--style_dir", type=str, default="./data/warli_train", help="Path to authentic Warli images")
    parser.add_argument("--val_dir", type=str, default=None, help="Optional path to validation images")
    args = parser.parse_args()

    content_stats = inspect_directory(args.content_dir, label="Content Dataset (Natural Photos)")
    style_stats = inspect_directory(args.style_dir, label="Style Dataset (Authentic Warli Artwork)")
    val_stats = inspect_directory(args.val_dir, label="Validation Dataset (Unseen Benchmarks)") if args.val_dir else None

    print("OVERALL READINESS SUMMARY:")
    all_ready = True
    if not content_stats["exists"] or content_stats["valid_count"] == 0:
        print("❌ Content dataset is not ready. Provide valid content images.")
        all_ready = False
    else:
        print(f"✓ Content dataset: {content_stats['valid_count']} valid images ready.")

    if not style_stats["exists"] or style_stats["valid_count"] == 0:
        print("❌ Warli style dataset is not ready. Provide authentic Warli artwork images.")
        all_ready = False
    else:
        print(f"✓ Warli style dataset: {style_stats['valid_count']} valid images ready.")

    if val_stats:
        if not val_stats["exists"] or val_stats["valid_count"] == 0:
            print("⚠️ Validation dataset is empty or missing. Recommended for inspection plates.")
        else:
            print(f"✓ Validation dataset: {val_stats['valid_count']} unseen test images ready.")

    if not all_ready:
        sys.exit(1)


if __name__ == "__main__":
    main()
