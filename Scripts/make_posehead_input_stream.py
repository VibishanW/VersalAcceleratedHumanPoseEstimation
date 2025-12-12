#!/usr/bin/env python3
import os
import json
import numpy as np

# Where things live
WEIGHTS_DIR = "weights"
GOLDEN_DIR = "golden_outputs"
STREAM_DIR = os.path.join(WEIGHTS_DIR, "streams")

# Q15 scale (same idea as weights)
Q_SCALE = 2**15  # 32768.0


def load_head_params(head_name: str):
    """
    Load FP32 weights/bias for a given head from the manifest.
    Returns (W_flat, b), with W_flat shape (out_ch, 1152).
    """
    manifest_path = os.path.join(WEIGHTS_DIR, "posehead_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    info = manifest["heads"][head_name]
    w_fp32 = np.load(os.path.join(WEIGHTS_DIR, info["w_fp32"]))  # (out_ch,2,2,288)
    b_fp32 = np.load(os.path.join(WEIGHTS_DIR, info["b_fp32"]))  # (out_ch,)

    out_ch = info["out_ch"]
    in_ch = info["in_ch"]  # should be 1152
    kh, kw = info["kernel"]

    # Sanity checks
    assert w_fp32.shape == (out_ch, kh, kw, in_ch // (kh * kw)), \
        f"{head_name}: unexpected weight shape {w_fp32.shape}"
    assert b_fp32.shape == (out_ch,), f"{head_name}: unexpected bias shape {b_fp32.shape}"

    # Flatten conv weights into FC weights: (out_ch, 2*2*288) = (out_ch, 1152)
    W_flat = w_fp32.reshape(out_ch, -1)  # row-major [h,w,c] => feature dim

    if W_flat.shape[1] != in_ch:
        raise RuntimeError(
            f"{head_name}: flattened weight second dim {W_flat.shape[1]} != in_ch {in_ch}"
        )

    return W_flat.astype(np.float32), b_fp32.astype(np.float32), info


def main():
    print("[1] Loading head weights, biases, and golden outputs…")

    # Load pose3d
    W_pose3d, b_pose3d, info_pose3d = load_head_params("pose3d")
    y_pose3d = np.load(os.path.join(GOLDEN_DIR, "golden_pose3d_fp32.npy")).reshape(-1)

    # Load world
    W_world, b_world, info_world = load_head_params("world")
    y_world = np.load(os.path.join(GOLDEN_DIR, "golden_world_fp32.npy")).reshape(-1)

    # Load flag
    W_flag, b_flag, info_flag = load_head_params("flag")
    y_flag = np.load(os.path.join(GOLDEN_DIR, "golden_flag_fp32.npy")).reshape(-1)

    print(f"    pose3d: W {W_pose3d.shape}, b {b_pose3d.shape}, y {y_pose3d.shape}")
    print(f"    world : W {W_world.shape},  b {b_world.shape},  y {y_world.shape}")
    print(f"    flag  : W {W_flag.shape},   b {b_flag.shape},   y {y_flag.shape}")

    # Build the big linear system A x = rhs by stacking all heads
    print("\n[2] Building joint linear system across all heads…")

    A = np.vstack([W_pose3d, W_world, W_flag])  # shape (195+117+1, 1152)
    rhs = np.concatenate([
        y_pose3d - b_pose3d,
        y_world  - b_world,
        y_flag   - b_flag,
    ])

    print(f"    A shape  : {A.shape}")
    print(f"    rhs shape: {rhs.shape}")

    print("\n[3] Solving for backbone feature vector x (least-squares)…")
    x, residuals, rank, s = np.linalg.lstsq(A, rhs, rcond=None)

    print(f"    x shape        : {x.shape} (expected 1152)")
    print(f"    rank(A)        : {rank}")
    if residuals.size > 0:
        print(f"    sum residual^2 : {residuals[0]:.6e}")
    else:
        print("    residuals      : (none reported by lstsq)")

    # Reshape to 2x2x288 feature map (for your own sanity checks)
    feat_fp32 = x.reshape(2, 2, 288)
    print(f"    reshaped feat  : {feat_fp32.shape} (2,2,288)")

    # Save FP32 feature
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    np.save(os.path.join(WEIGHTS_DIR, "posehead_backbone_feat_fp32.npy"), feat_fp32)
    print("    -> saved FP32 feature as weights/posehead_backbone_feat_fp32.npy")

    # Quantize to Q15
    print("\n[4] Quantizing feature to Q15 and building PLIO stream…")
    feat_q15 = np.round(feat_fp32 * Q_SCALE).astype(np.int32)
    feat_q15 = np.clip(feat_q15, -32768, 32767)

    # Save as .npy for debugging
    np.save(os.path.join(WEIGHTS_DIR, "posehead_backbone_feat_q15.npy"), feat_q15)
    print("    -> saved Q15 feature as weights/posehead_backbone_feat_q15.npy")

    # Flatten in [h,w,c] row-major order (same as reshape)
    stream_1d = feat_q15.reshape(-1)  # length 1152

    os.makedirs(STREAM_DIR, exist_ok=True)
    stream_path = os.path.join(STREAM_DIR, "posehead_input_stream.txt")
    with open(stream_path, "w") as f:
        for v in stream_1d:
            f.write(f"{int(v)}\n")

    print(f"    -> wrote PLIO input stream: {stream_path} (len={len(stream_1d)})")

    print("\n[5] Sanity-check: re-evaluate heads with recovered feature…")

    # Recompute y = W x + b and compare to golden
    x_vec = feat_fp32.reshape(-1)  # 1152

    def head_check(name, W, b, y_gold):
        y_hat = W @ x_vec + b
        diff = y_hat - y_gold
        max_abs = np.max(np.abs(diff))
        rms = np.sqrt(np.mean(diff**2))
        print(f"    {name}: max_abs_err={max_abs:.6e}, rms_err={rms:.6e}")

    head_check("pose3d", W_pose3d, b_pose3d, y_pose3d)
    head_check("world ", W_world,  b_world,  y_world)
    head_check("flag  ", W_flag,   b_flag,   y_flag)

    print("\n[DONE] Pose-head input PLIO stream generated.")
    print("       Use weights/streams/posehead_input_stream.txt as your feat_in PLIO source.")
    print("       (Or copy/rename it to match your graph's expected path, e.g. data/in_stream.txt.)")


if __name__ == "__main__":
    main()
