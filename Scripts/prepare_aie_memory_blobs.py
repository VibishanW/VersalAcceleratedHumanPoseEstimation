#!/usr/bin/env python3
import os
import numpy as np

# ------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------
# Base directory = folder containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the original text streams live and where we will write all outputs
STREAM_DIR = os.path.join(BASE_DIR, "weights", "streams")

# Correct channel counts from TFLite
POSE3D_IN_CH = 1152  # 2×2×288
WORLD_IN_CH  = 1152  # 2×2×288
FLAG_IN_CH   = 1152  # 2×2×288 


# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def read_int16_stream(path: str) -> np.ndarray:
    """Read a plain-text stream of ints as int16 numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.loadtxt(path, dtype=np.int16)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def write_npy_and_bin(base_path: str, arr: np.ndarray):
    """Save both .npy and raw .bin (little-endian int16)."""
    npy_path = base_path + ".npy"
    bin_path = base_path + ".bin"
    np.save(npy_path, arr)
    arr.astype(np.int16).tofile(bin_path)
    print(f"  -> {npy_path} (shape {arr.shape}, dtype={arr.dtype})")
    print(f"  -> {bin_path} (raw {arr.size} int16 elements)")


def write_plio_txt_int16(arr: np.ndarray, txt_path: str):
    """
    Write a 1D int16 array to a PLIO-style text file:
    - 8 samples per line
    - pad with zeros to multiple of 8 if needed
    """
    flat = arr.reshape(-1).astype(np.int16)
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int16)])
        print(f"  [PLIO] padded {pad} zeros → total {flat.size} samples for {os.path.basename(txt_path)}")

    with open(txt_path, "w") as f:
        for i in range(0, flat.size, 8):
            chunk = flat[i:i+8]
            f.write(" ".join(str(int(v)) for v in chunk) + "\n")
    print(f"  -> {txt_path} (PLIO text, 8 samples/line)")


# ------------------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------------------
def build_input_blob():
    """
    Shared pose-head input:
      - Reads posehead_input_stream.txt (int16)
      - Writes posehead_input_q15.npy/.bin
      - Writes PLIO feature streams: pose3d_feat.txt, world_feat.txt, flag_feat.txt
    """
    print("[1] Preparing shared pose-head input blob from posehead_input_stream.txt")

    in_path = os.path.join(STREAM_DIR, "posehead_input_stream.txt")
    x = read_int16_stream(in_path)
    print(f"  posehead_input_stream.txt length: {x.size}")

    # Binary blob (for host/AIE if needed)
    base = os.path.join(STREAM_DIR, "posehead_input_q15")
    write_npy_and_bin(base, x)

    # PLIO feature streams (all three heads share the same backbone features)
    for name in ["pose3d_feat.txt", "world_feat.txt", "flag_feat.txt"]:
        txt_path = os.path.join(STREAM_DIR, name)
        write_plio_txt_int16(x, txt_path)


def build_head_blobs(
    head_name: str,
    in_ch: int,
    w_stream_name: str,
    b_stream_name: str,
    expected_out_ch: int = None,
):
    """
    For a given head (pose3d / world / flag):
      - Read weights and biases streams (.txt)
      - Infer OUT_CH from weight length
      - Save:
          head_weights_q15.{npy,bin}     → weights [OUT_CH, IN_CH]
          head_bias_q15.{npy,bin}        → bias    [OUT_CH]
          head_fc_q15.bin                → packed FC rows [OUT_CH, 1+IN_CH]
          head_w.txt                     → PLIO text stream (bias + weights, 8 samples/line)
    """
    print(f"\n[+] Preparing {head_name} head blobs")

    w_txt = os.path.join(STREAM_DIR, w_stream_name)
    b_txt = os.path.join(STREAM_DIR, b_stream_name)

    w_flat = read_int16_stream(w_txt)
    b      = read_int16_stream(b_txt)

    print(f"  raw weights len = {w_flat.size}")
    print(f"  raw bias len    = {b.size}")

    if w_flat.size % in_ch != 0:
        raise ValueError(
            f"{head_name}: weights length {w_flat.size} not divisible by IN_CH={in_ch}"
        )

    out_ch = w_flat.size // in_ch
    print(f"  IN_CH  = {in_ch}")
    print(f"  OUT_CH = {out_ch} (inferred from stream length)")

    if expected_out_ch is not None and out_ch != expected_out_ch:
        raise ValueError(
            f"{head_name}: expected OUT_CH={expected_out_ch}, got {out_ch} "
            f"(check stream or IN_CH)"
        )

    if b.size != out_ch:
        raise ValueError(
            f"{head_name}: bias length {b.size} does not match OUT_CH={out_ch}"
        )

    # Reshape weights to (OUT_CH, IN_CH)
    w_mat = w_flat.reshape(out_ch, in_ch)
    print(f"  weights matrix shape: {w_mat.shape}")
    print(f"  bias vector shape   : {b.shape}")

    # Save plain weights/bias blobs
    w_base = os.path.join(STREAM_DIR, f"{head_name}_weights_q15")
    b_base = os.path.join(STREAM_DIR, f"{head_name}_bias_q15")
    write_npy_and_bin(w_base, w_mat)
    write_npy_and_bin(b_base, b)

    # Build FC-packed blob: [bias, w0..w_IN-1] per output row
    fc = np.zeros((out_ch, 1 + in_ch), dtype=np.int16)
    fc[:, 0]  = b
    fc[:, 1:] = w_mat

    print(f"  FC packed shape: {fc.shape} → {fc.size} int16 values")
    fc_base = os.path.join(STREAM_DIR, f"{head_name}_fc_q15")
    fc.tofile(fc_base + ".bin")
    print(f"  -> {fc_base}.bin")

    # PLIO text stream for weights+bias (what AIE PLIO will read)
    # Flatten row-major: bias0, w0_0..w0_(IN-1), bias1, w1_0.. etc.
    flat_fc = fc.reshape(-1)
    txt_name = f"{head_name}_w.txt"  # matches pose3d_w.txt etc.
    txt_path = os.path.join(STREAM_DIR, txt_name)
    write_plio_txt_int16(flat_fc, txt_path)


def main():
    print(f"[0] STREAM_DIR = {STREAM_DIR}")

    # 1) Shared input features
    build_input_blob()

    # 2) Pose3D head: 1152 → 195
    build_head_blobs(
        head_name       = "pose3d",
        in_ch           = POSE3D_IN_CH,
        w_stream_name   = "pose3d_head_weights_stream.txt",
        b_stream_name   = "pose3d_head_bias_stream.txt",
        expected_out_ch = 195,
    )

    # 3) World head: 1152 → 117
    build_head_blobs(
        head_name       = "world",
        in_ch           = WORLD_IN_CH,
        w_stream_name   = "world_head_weights_stream.txt",
        b_stream_name   = "world_head_bias_stream.txt",
        expected_out_ch = 117,
    )

    # 4) Flag head: 1152 → 1
    build_head_blobs(
        head_name       = "flag",
        in_ch           = FLAG_IN_CH,
        w_stream_name   = "flag_head_weights_stream.txt",
        b_stream_name   = "flag_head_bias_stream.txt",
        expected_out_ch = 1,
    )

    print("\n[DONE] All pose-head blobs and PLIO streams built successfully.")


if __name__ == "__main__":
    main()