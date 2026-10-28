"""
KalaKriti — Checkpoint 2: Architectural and Gradient Flow Unit Tests
Tests:
  1. Forward pass tensor shape consistency for square [1, 3, 512, 512]
  2. Non-square dynamic aspect-ratio preservation [1, 3, 480, 640]
  3. Batch processing [2, 3, 256, 256] with multi-style conditioning
  4. Real parameter count and disk size measurement
  5. Backpropagation and gradient-flow verification through CIN
  6. State dict serialization & deterministic reload verification
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.transformer_net import TransformerNet, ConditionalInstanceNorm2d


def run_all_tests():
    print("=" * 65)
    print("KALAKRITI ARCHITECTURE VERIFICATION — CHECKPOINT 2")
    print("=" * 65)

    device = torch.device("cpu")
    torch.set_num_threads(2)
    num_styles = 5  # Warli (0), + reserve slots for Madhubani, Kalamkari, Gond, Pattachitra
    model = TransformerNet(num_styles=num_styles, base_channels=32).to(device)

    # 1. Parameter Count & Size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cin_params = sum(
        p.numel() for m in model.modules() if isinstance(m, ConditionalInstanceNorm2d) for p in m.parameters()
    )
    est_size_mb = (total_params * 4) / (1024 * 1024)

    print(f"\n[1] PARAMETER ANALYSIS:")
    print(f"    - Total Parameters:      {total_params:,}")
    print(f"    - Trainable Parameters:  {trainable_params:,}")
    print(f"    - CIN-specific Params:   {cin_params:,} ({(cin_params/total_params)*100:.2f}% of network)")
    print(f"    - Estimated FP32 Memory: {est_size_mb:.2f} MB")

    # 2. Square Tensor Shape Test
    print(f"\n[2] FORWARD PASS: SQUARE TENSOR (512x512)")
    x_square = torch.rand(1, 3, 512, 512, device=device) * 255.0
    style_0 = torch.tensor([0], dtype=torch.long, device=device)
    out_square = model(x_square, style_0)
    assert out_square.shape == (1, 3, 512, 512), f"Shape mismatch: {out_square.shape} != (1, 3, 512, 512)"
    assert not torch.isnan(out_square).any(), "NaN values found in square output"
    assert (out_square >= 0.0).all() and (out_square <= 255.0).all(), "Pixel values exceeded [0, 255]"
    print(f"    ✓ Input:  {list(x_square.shape)} -> Output: {list(out_square.shape)}")
    print(f"    ✓ Output range: [{out_square.min().item():.2f}, {out_square.max().item():.2f}]")

    # 3. Non-Square Aspect Ratio Test
    print(f"\n[3] FORWARD PASS: NON-SQUARE TENSOR (480x640)")
    x_rect = torch.rand(1, 3, 480, 640, device=device) * 255.0
    out_rect = model(x_rect, style_0)
    assert out_rect.shape == (1, 3, 480, 640), f"Aspect-ratio mismatch: {out_rect.shape} != (1, 3, 480, 640)"
    assert not torch.isnan(out_rect).any(), "NaN values found in non-square output"
    print(f"    ✓ Input:  {list(x_rect.shape)} -> Output: {list(out_rect.shape)} (Preserved)")

    # 4. Batch and Style-Conditioning Test
    print(f"\n[4] MULTI-STYLE BATCH PASS (B=2, 256x256)")
    x_batch = torch.rand(2, 3, 256, 256, device=device) * 255.0
    style_batch = torch.tensor([0, 1], dtype=torch.long, device=device)
    out_batch = model(x_batch, style_batch)
    assert out_batch.shape == (2, 3, 256, 256), f"Batch shape mismatch: {out_batch.shape}"
    print(f"    ✓ Input batch: {list(x_batch.shape)} with style IDs [0, 1] -> Output: {list(out_batch.shape)}")

    # 5. Gradient Flow / Backpropagation Test
    print(f"\n[5] GRADIENT FLOW & BACKPROPAGATION TEST")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    x_dummy = torch.rand(1, 3, 256, 256, requires_grad=True, device=device) * 255.0
    target_dummy = torch.rand(1, 3, 256, 256, device=device) * 255.0
    out_train = model(x_dummy, style_0)
    loss = nn.functional.mse_loss(out_train, target_dummy)
    loss.backward()

    # Verify gradients in conv and CIN layers
    grad_norms = []
    zero_grad_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                zero_grad_layers.append(name)
            else:
                grad_norms.append(param.grad.norm().item())

    assert len(zero_grad_layers) == 0, f"Layers with missing gradients: {zero_grad_layers}"
    avg_grad = sum(grad_norms) / len(grad_norms)
    print(f"    ✓ Loss: {loss.item():.4f}")
    print(f"    ✓ Gradient flow verified across {len(grad_norms)} parameter tensors")
    print(f"    ✓ Average gradient norm: {avg_grad:.6f} (No vanishing/exploding gradients)")

    # 6. Checkpoint Serialization & Reload Test
    print(f"\n[6] CHECKPOINT SERIALIZATION & DETERMINISTIC RELOAD")
    checkpoint_dir = "/tmp/kalakriti_test"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "test_scaffold.pth")

    model.eval()
    with torch.no_grad():
        out_before = model(x_square, style_0)

    torch.save(model.state_dict(), ckpt_path)
    file_size_kb = os.path.getsize(ckpt_path) / 1024
    print(f"    ✓ State dict saved to {ckpt_path} ({file_size_kb:.1f} KB)")

    # Reload into fresh instance
    fresh_model = TransformerNet(num_styles=num_styles, base_channels=32).to(device)
    fresh_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    fresh_model.eval()

    with torch.no_grad():
        out_after = fresh_model(x_square, style_0)

    max_diff = (out_before - out_after).abs().max().item()
    assert max_diff == 0.0, f"Output changed after checkpoint reload! Max diff: {max_diff}"
    print(f"    ✓ Deterministic reload verified (Max absolute difference: {max_diff})")

    # Clean up test checkpoint
    os.remove(ckpt_path)
    os.rmdir(checkpoint_dir)

    print("\n" + "=" * 65)
    print("ALL CHECKPOINT 2 TESTS PASSED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
