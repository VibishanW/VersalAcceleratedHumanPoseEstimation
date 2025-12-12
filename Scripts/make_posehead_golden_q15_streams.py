#!/usr/bin/env python3
import os
import numpy as np

GOLDEN_DIR = "golden_outputs"
WEIGHTS_DIR = "weights"
STREAM_DIR = os.path.join(WEIGHTS_DIR, "streams")

Q_SCALE = 2**15  # Q15


def quantize_to_q15(x: np.ndarray) -> np.ndarray:
    """Quantize float32 array to Q15 int16 (returned as int32 for PLIO friendliness)."""
    x = x.astype(np.float32)
    q = np.round(x * Q_SCALE).astype(np.int32)
    q = np.clip(q, -32768, 32767)
    return q


def save_head_stream(head_name: str,
                     golden_fname_fp32: str,
                     npy_q15_name: str,
                     stream_txt_name: str):
    """Load FP32 golden, quantize to Q15, save .npy and text stream."""
    path_fp32 = os.path.join(GOLDEN_DIR, golden_fname_fp32)
    if not os.path.exists(path_fp32):
        raise FileNotFoundError(f"Missing {path_fp32}")

    y_fp32 = np.load(path_fp32)
    print(f"  {head_name}: FP32 {golden_fname_fp32} shape={y_fp32.shape}, dtype={y_fp32.dtype}")

    # Flatten to 1D (but keep full content)
    y_flat = y_fp32.reshape(-1)
    y_q15 = quantize_to_q15(y_flat)

    # Save Q15 .npy in golden_outputs
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    npy_q15_path = os.path.join(GOLDEN_DIR, npy_q15_name)
    np.save(npy_q15_path, y_q15.astype(np.int16))
    print(f"    -> saved Q15 NPY: {npy_q15_path} (shape={y_q15.shape})")

    # Save text stream in weights/streams
    os.makedirs(STREAM_DIR, exist_ok=True)
    stream_path = os.path.join(STREAM_DIR, stream_txt_name)
    with open(stream_path, "w") as f:
        for v in y_q15:
            f.write(f"{int(v)}\n")
    print(f"    -> saved text stream: {stream_path} (len={len(y_q15)})")

    # Simple stats
    print(f"    Q15 stats: min={y_q15.min()}, max={y_q15.max()}")


def main():
    print("[1] Generating Q15 golden output streams for pose-heads…")

    # Pose3D (1,195)
    save_head_stream(
        head_name="pose3d",
        golden_fname_fp32="golden_pose3d_fp32.npy",
        npy_q15_name="golden_pose3d_q15.npy",
        stream_txt_name="pose3d_golden_out_q15.txt",
    )

    # World (1,117)
    save_head_stream(
        head_name="world",
        golden_fname_fp32="golden_world_fp32.npy",
        npy_q15_name="golden_world_q15.npy",
        stream_txt_name="world_golden_out_q15.txt",
    )

    # Flag (1,1)
    save_head_stream(
        head_name="flag",
        golden_fname_fp32="golden_flag_fp32.npy",
        npy_q15_name="golden_flag_q15.npy",
        stream_txt_name="flag_golden_out_q15.txt",
    )

    print("\n[DONE] Q15 golden output streams created.")
    print("       Compare your AIE kernel outputs against these streams line-by-line.")
    print("       All text streams are under: weights/streams/")


if __name__ == "__main__":
    main()
