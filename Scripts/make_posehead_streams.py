#!/usr/bin/env python3
import os
import json
import numpy as np

WEIGHTS_DIR = "weights"
OUT_DIR = os.path.join(WEIGHTS_DIR, "streams")


def write_stream_txt(path, arr_1d):
    """
    Write a 1D array of ints to a text file, one per line.
    Values are written as Python ints (so effectively int32).
    """
    with open(path, "w") as f:
        for v in arr_1d:
            f.write(f"{int(v)}\n")


def main():
    print("[1] Loading manifest and Q15 weights…")
    manifest_path = os.path.join(WEIGHTS_DIR, "posehead_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    heads = ["pose3d", "world", "flag"]

    for head in heads:
        info = manifest["heads"][head]
        out_ch = info["out_ch"]
        in_ch = info["in_ch"]  # should be 1152
        kernel_h, kernel_w = info["kernel"]

        print(f"\n=== {head.upper()} ===")
        print(f"  out_ch={out_ch}, in_ch={in_ch}, kernel={kernel_h}x{kernel_w}")

        # Load Q15 weights & biases
        w_q15_path = os.path.join(WEIGHTS_DIR, info["w_q15"])
        b_q15_path = os.path.join(WEIGHTS_DIR, info["b_q15"])

        w_q15 = np.load(w_q15_path)   # shape: (out_ch, 2, 2, 288)
        b_q15 = np.load(b_q15_path)   # shape: (out_ch,) or (1,) for flag

        print(f"  w_q15 shape: {w_q15.shape}, dtype={w_q15.dtype}")
        print(f"  b_q15 shape: {b_q15.shape}, dtype={b_q15.dtype}")

        # Flatten conv weights into FC weights: (out_ch, 2*2*288) = (out_ch, 1152)
        w_flat = w_q15.reshape(out_ch, -1)  # row-major: [h,w,c]
        if w_flat.shape[1] != in_ch:
            raise RuntimeError(
                f"{head}: flattened weight second dimension {w_flat.shape[1]} != in_ch {in_ch}"
            )

        print(f"  w_flat shape: {w_flat.shape}")

        # Make 1D streams by concatenating rows in order:
        #   weights_stream = [w[0,0], w[0,1], ..., w[0,1151],
        #                     w[1,0], ..., w[1,1151],
        #                     ...]
        w_stream = w_flat.astype(np.int32).reshape(-1)

        # Bias: just in out_ch order (or 1 for flag)
        b_stream = b_q15.astype(np.int32).reshape(-1)

        # Write to text files
        w_stream_path = os.path.join(OUT_DIR, f"{head}_head_weights_stream.txt")
        b_stream_path = os.path.join(OUT_DIR, f"{head}_head_bias_stream.txt")

        write_stream_txt(w_stream_path, w_stream)
        write_stream_txt(b_stream_path, b_stream)

        print(f"  -> wrote weights stream: {w_stream_path} (len={len(w_stream)})")
        print(f"  -> wrote bias    stream: {b_stream_path} (len={len(b_stream)})")

    print("\n[DONE] Pose-head streams written to:", OUT_DIR)


if __name__ == "__main__":
    main()
