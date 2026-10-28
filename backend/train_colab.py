"""
KalaKriti — Real-Time Multi-Style Neural Style Transfer
Google Colab Training Pipeline (Stage 5A Pilot & Stage 5 Full Training)
Target Art Style: Warli Traditional Art (Style ID: 0)

Key Enhancements:
  - Multi-artwork Gram matrix averaging (learns the collective style domain, not just 1 image)
  - Fixed random seeds for reproducible pilot runs
  - Unseen validation benchmark evaluator (saves side-by-side comparison plates)
  - Separate loss logging (content, multi-scale style, TV, total)
"""

import argparse
import os
import random
import sys
import time
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.models as models

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.models.transformer_net import TransformerNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VGG19FeatureExtractor(nn.Module):
    """
    Pretrained VGG-19 with AvgPool substitution for smooth gradient flow.
    Extracts:
      - Content: relu4_2 (semantic pose/structure anchor)
      - Style: relu1_2, relu2_2, relu3_3, relu4_3 (hierarchical stroke & motif capture)
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Replace MaxPool2d with AvgPool2d to prevent gradient dropouts
        layers = []
        for layer in vgg:
            if isinstance(layer, nn.MaxPool2d):
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(layer)
        self.features = nn.Sequential(*layers)

        # Freeze all VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        # x is in [0, 255]
        return (x / 255.0 - self.mean) / self.std

    def forward(self, x):
        h = self.normalize(x)
        features = {}
        for idx, layer in enumerate(self.features):
            h = layer(h)
            if idx == 3:   # relu1_2
                features["relu1_2"] = h
            elif idx == 8:  # relu2_2
                features["relu2_2"] = h
            elif idx == 15: # relu3_3
                features["relu3_3"] = h
            elif idx == 22: # relu4_2 (Content layer)
                features["relu4_2"] = h
            elif idx == 24: # relu4_3 (High-level style)
                features["relu4_3"] = h
            if idx > 24:
                break
        return features


def gram_matrix(y):
    """Computes normalized Gram matrix for style feature representations."""
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def total_variation_loss(img):
    """Computes total variation loss to suppress high-frequency noise."""
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


class ImageFolderDataset(Dataset):
    """Loads image files from a directory and applies standard transforms."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        if os.path.exists(root_dir):
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    self.image_paths.append(os.path.join(root_dir, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(path)


def compute_domain_averaged_gram_matrices(vgg, style_loader, device, batch_size):
    """
    Averages Gram matrices across the entire authentic Warli training collection.
    Prevents overfitting to a single painting and learns general Warli geometric features.
    """
    print(f"Computing domain-averaged Gram matrices across {len(style_loader.dataset)} Warli artworks...")
    accumulated_grams = {}
    style_count = 0

    with torch.no_grad():
        for style_img, _ in style_loader:
            style_img = style_img.to(device)
            feats = vgg(style_img)
            for k, v in feats.items():
                if k != "relu4_2":
                    g = gram_matrix(v)
                    if k not in accumulated_grams:
                        accumulated_grams[k] = g
                    else:
                        accumulated_grams[k] += g
            style_count += 1

    # Average and repeat across batch dimension
    target_grams = {}
    for k, v in accumulated_grams.items():
        avg_g = v / max(1, style_count)
        target_grams[k] = avg_g.repeat(batch_size, 1, 1)

    print(f"✓ Domain Gram matrices computed for layers: {list(target_grams.keys())}")
    return target_grams


def run_visual_validation(generator, vgg, val_dir, style_ref_path, output_dir, device, epoch, step):
    """
    Renders unseen validation images (portrait, monument, landscape, animal)
    alongside the reference Warli artwork to produce side-by-side inspection plates.
    """
    os.makedirs(output_dir, exist_ok=True)
    val_files = [f for f in sorted(os.listdir(val_dir)) if f.lower().endswith((".jpg", ".png", ".jpeg"))] if os.path.exists(val_dir) else []
    if not val_files:
        return

    # Load style reference thumbnail
    style_ref_thumb = None
    if os.path.exists(style_ref_path):
        with Image.open(style_ref_path) as s_img:
            style_ref_thumb = s_img.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)

    generator.eval()
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    print(f"\n--- Generating Unseen Validation Inspection Plate (Epoch {epoch}, Step {step}) ---")
    with torch.no_grad():
        for fname in val_files[:4]:  # Evaluate up to 4 unseen benchmark categories
            fpath = os.path.join(val_dir, fname)
            raw_img = Image.open(fpath).convert("RGB")
            content_tensor = val_transform(raw_img).unsqueeze(0).to(device)
            style_id = torch.tensor([0], dtype=torch.long, device=device)

            out_tensor = generator(content_tensor, style_id)
            out_tensor = torch.clamp(out_tensor, 0.0, 255.0).byte()

            # Prepare comparison plate: [Original Content | Warli Reference | KalaKriti Stylized]
            content_np = content_tensor.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
            out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

            content_pil = Image.fromarray(content_np)
            out_pil = Image.fromarray(out_np)

            plate_w = 256 * 3
            plate = Image.new("RGB", (plate_w, 256))
            plate.paste(content_pil, (0, 0))
            if style_ref_thumb:
                plate.paste(style_ref_thumb, (256, 0))
            plate.paste(out_pil, (512, 0))

            save_name = f"val_epoch{epoch}_step{step}_{os.path.splitext(fname)[0]}_comparison.jpg"
            save_path = os.path.join(output_dir, save_name)
            plate.save(save_path, quality=90)
            print(f"  Saved comparison plate: {save_path}")

    generator.train()


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting KalaKriti training pipeline on device: {device} (Seed: {args.seed})")

    # 1. Transform Pipelines
    content_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    style_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    # 2. Dataset Setup
    content_dataset = ImageFolderDataset(args.content_dir, transform=content_transform)
    warli_dataset = ImageFolderDataset(args.style_dir, transform=style_transform)

    if len(content_dataset) == 0:
        print(f"❌ Error: No content images found in '{args.content_dir}'.")
        return
    if len(warli_dataset) == 0:
        print(f"❌ Error: No authentic Warli images found in '{args.style_dir}'.")
        return

    # Pilot Mode Subset Constraint
    if args.pilot and args.max_samples > 0:
        actual_samples = min(args.max_samples, len(content_dataset))
        content_dataset = Subset(content_dataset, range(actual_samples))
        print(f"🧪 PILOT EXPERIMENT MODE ENABLED: Using {actual_samples} content images.")

    content_loader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    style_loader = DataLoader(warli_dataset, batch_size=1, shuffle=False)

    print(f"Loaded {len(content_dataset)} content training images.")
    print(f"Loaded {len(warli_dataset)} authentic Warli style reference images.")

    # 3. Model & Feature Extractor Initialization
    generator = TransformerNet(num_styles=5, base_channels=32).to(device)
    vgg = VGG19FeatureExtractor().to(device).eval()
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

    # 4. Compute Domain-Averaged Target Gram Matrices
    target_grams = compute_domain_averaged_gram_matrices(vgg, style_loader, device, args.batch_size)

    # Calibrated Layer Weights (up-weighting relu3_3 and relu4_3 to capture stroke and geometric motif structures)
    layer_weights = {
        "relu1_2": 0.1,
        "relu2_2": 0.3,
        "relu3_3": 1.0,
        "relu4_3": 1.0,
    }

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    step = 0
    start_time = time.time()

    # Pick first style image as visual reference for comparison plates
    first_style_ref = warli_dataset.image_paths[0] if len(warli_dataset.image_paths) > 0 else ""

    print(f"\n--- Starting Training Loop ({args.epochs} Epochs) ---")
    generator.train()
    for epoch in range(args.epochs):
        for batch_idx, (content_batch, _) in enumerate(content_loader):
            step += 1
            content_batch = content_batch.to(device)
            style_id = torch.zeros(args.batch_size, dtype=torch.long, device=device)

            optimizer.zero_grad()

            # Forward pass through generator
            output_batch = generator(content_batch, style_id)

            # Perceptual feature extraction
            out_feats = vgg(output_batch)
            content_feats = vgg(content_batch)

            # A. Content Loss (anchored at relu4_2)
            content_loss = args.content_weight * nn.functional.mse_loss(
                out_feats["relu4_2"], content_feats["relu4_2"]
            )

            # B. Hierarchical Style Loss
            style_loss = 0.0
            for layer_name, weight in layer_weights.items():
                if layer_name in out_feats and layer_name in target_grams:
                    out_gram = gram_matrix(out_feats[layer_name])
                    style_loss += weight * nn.functional.mse_loss(out_gram, target_grams[layer_name])
            style_loss *= args.style_weight

            # C. Total Variation Regularization
            tv_loss = args.tv_weight * total_variation_loss(output_batch)

            # D. Total Combined Loss
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()

            if step % args.log_interval == 0:
                elapsed = time.time() - start_time
                gpu_mem = f"{torch.cuda.memory_allocated() / (1024**2):.1f} MB" if torch.cuda.is_available() else "CPU"
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Step [{step}] | "
                    f"Content: {content_loss.item():.4f} | Style: {style_loss.item():.4f} | "
                    f"TV: {tv_loss.item():.4f} | Total: {total_loss.item():.4f} | Mem: {gpu_mem} | Time: {elapsed:.1f}s"
                )

        # Generate visual inspection plate at end of epoch
        if args.val_dir and os.path.exists(args.val_dir):
            run_visual_validation(
                generator, vgg, args.val_dir, first_style_ref, args.output_dir, device, epoch + 1, step
            )

        # Save checkpoint per epoch
        ckpt_suffix = "_pilot" if args.pilot else ""
        epoch_path = os.path.join(args.checkpoint_dir, f"kalakriti_cin_warli{ckpt_suffix}_epoch{epoch+1}.pth")
        torch.save(generator.state_dict(), epoch_path)
        print(f"Saved epoch checkpoint to {epoch_path}")

    # Save final operational weights
    final_filename = "kalakriti_cin_warli_pilot.pth" if args.pilot else "kalakriti_cin_warli.pth"
    final_path = os.path.join(args.checkpoint_dir, final_filename)
    torch.save(generator.state_dict(), final_path)
    print(f"\nTraining session complete. Final weights saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KalaKriti Multi-Style Generator")
    parser.add_argument("--content_dir", type=str, default="./data/coco_train")
    parser.add_argument("--style_dir", type=str, default="./data/warli_train")
    parser.add_argument("--val_dir", type=str, default="./data/val_content", help="Unseen test images for validation")
    parser.add_argument("--checkpoint_dir", type=str, default="./backend/checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output/val_plates")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--content_weight", type=float, default=1.0)
    parser.add_argument("--style_weight", type=float, default=5e3)
    parser.add_argument("--tv_weight", type=float, default=1e-5)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true", help="Run pilot mode on small subset")
    parser.add_argument("--max_samples", type=int, default=100, help="Max content samples for pilot")

    args = parser.parse_args()
    train(args)
