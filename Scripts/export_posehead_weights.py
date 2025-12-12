#!/usr/bin/env python3
import os
import numpy as np

# ---------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------
MODEL_PATH = "pose_landmark_full.tflite"
OUT_DIR    = "weights"

# Disable XNNPACK to avoid any weirdness with internal tensors
os.environ.setdefault("TFLITE_DISABLE_XNNPACK", "1")

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback, in case you ever have full TF installed instead
    import tensorflow as tf
    tflite = tf.lite


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def quantize_q15(arr: np.ndarray) -> np.ndarray:
    """
    Convert float (FP16/FP32) to Q15 int16: x_q15 = round(x * 2^15), clamped.
    """
    a = arr.astype(np.float64)
    scale = float(1 << 15)
    q = np.round(a * scale)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q


def find_tensor(details, name_substr, expected_shape):
    """
    Find a unique tensor whose name contains `name_substr` and whose shape
    matches `expected_shape`. Prefer non-*_dequantize tensors.
    """
    matches = []
    for d in details:
        if name_substr in d["name"] and tuple(d["shape"]) == tuple(expected_shape):
            matches.append(d)

    if not matches:
        msg = [
            f"ERROR: No tensor found for substring='{name_substr}',",
            f"       expected shape={tuple(expected_shape)}",
        ]
        raise RuntimeError("\n".join(msg))

    # Prefer non-dequantize tensors
    non_deq = [d for d in matches if "dequantize" not in d["name"].lower()]
    if len(non_deq) == 1:
        return non_deq[0]
    if len(non_deq) > 1:
        matches = non_deq

    if len(matches) != 1:
        msg = ["ERROR: Ambiguous tensor match:"]
        for d in matches:
            msg.append(
                f"  idx={d['index']}, name={d['name']}, shape={tuple(d['shape'])}, dtype={d['dtype']}"
            )
        raise RuntimeError("\n".join(msg))

    return matches[0]


def dump_tensor(interpreter, tensor_info, base_name):
    """
    Get tensor data from interpreter and save:
      <OUT_DIR>/<base_name>_fp32.npy
      <OUT_DIR>/<base_name>_q15.npy
    Returns (fp32_array, q15_array).
    """
    idx = tensor_info["index"]
    arr = interpreter.get_tensor(idx)

    # Convert FP16 -> FP32 if needed
    if arr.dtype == np.float16:
        arr_fp32 = arr.astype(np.float32)
    else:
        arr_fp32 = arr.astype(np.float32)

    q15 = quantize_q15(arr_fp32)

    os.makedirs(OUT_DIR, exist_ok=True)
    fp32_path = os.path.join(OUT_DIR, f"{base_name}_fp32.npy")
    q15_path  = os.path.join(OUT_DIR, f"{base_name}_q15.npy")

    np.save(fp32_path, arr_fp32)
    np.save(q15_path, q15)

    print(f"    Saved {base_name}:")
    print(f"      FP32 -> {fp32_path}  shape={arr_fp32.shape}, dtype={arr_fp32.dtype}")
    print(f"      Q15  -> {q15_path}   shape={q15.shape}, dtype={q15.dtype}")

    return arr_fp32, q15


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    print(f"[1] Loading TFLite model: {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    print(f"    allocate_tensors() done, #tensors = {len(details)}")

    # -----------------------------------------------------------------
    #  Find pose-head tensors (names/shapes from tensors.json)
    # -----------------------------------------------------------------
    print("\n[2] Locating pose-head tensors (weights & biases)…")

    # Pose3D head
    pose3d_w_info = find_tensor(
        details,
        name_substr="model_1/model/convld_3d/Conv2D",
        expected_shape=(195, 2, 2, 288),
    )
    pose3d_b_info = find_tensor(
        details,
        name_substr="model_1/model/convld_3d/BiasAdd",
        expected_shape=(195,),
    )

    print("  Pose3D weights:",
          f"idx={pose3d_w_info['index']}, shape={tuple(pose3d_w_info['shape'])},",
          f"dtype={pose3d_w_info['dtype']}, name={pose3d_w_info['name']}")
    print("  Pose3D bias   :",
          f"idx={pose3d_b_info['index']}, shape={tuple(pose3d_b_info['shape'])},",
          f"dtype={pose3d_b_info['dtype']}, name={pose3d_b_info['name']}")

    # World head
    world_w_info = find_tensor(
        details,
        name_substr="model_1/model/convworld_3d/Conv2D",
        expected_shape=(117, 2, 2, 288),
    )
    world_b_info = find_tensor(
        details,
        name_substr="model_1/model/convworld_3d/BiasAdd",
        expected_shape=(117,),
    )

    print("  World  weights:",
          f"idx={world_w_info['index']}, shape={tuple(world_w_info['shape'])},",
          f"dtype={world_w_info['dtype']}, name={world_w_info['name']}")
    print("  World  bias   :",
          f"idx={world_b_info['index']}, shape={tuple(world_b_info['shape'])},",
          f"dtype={world_b_info['dtype']}, name={world_b_info['name']}")

    # Flag head
    flag_w_info = find_tensor(
        details,
        name_substr="model_1/model/conv_poseflag/Conv2D",
        expected_shape=(1, 2, 2, 288),
    )
    flag_b_info = find_tensor(
        details,
        name_substr="model_1/model/conv_poseflag/BiasAdd",
        expected_shape=(1,),
    )

    print("  Flag   weights:",
          f"idx={flag_w_info['index']}, shape={tuple(flag_w_info['shape'])},",
          f"dtype={flag_w_info['dtype']}, name={flag_w_info['name']}")
    print("  Flag   bias   :",
          f"idx={flag_b_info['index']}, shape={tuple(flag_b_info['shape'])},",
          f"dtype={flag_b_info['dtype']}, name={flag_b_info['name']}")

    # -----------------------------------------------------------------
    #  Dump to FP32 + Q15 .npy files
    # -----------------------------------------------------------------
    print("\n[3] Exporting weights/biases (FP32 + Q15)…")

    pose3d_w_fp32, pose3d_w_q15 = dump_tensor(interpreter, pose3d_w_info, "pose3d_w")
    pose3d_b_fp32, pose3d_b_q15 = dump_tensor(interpreter, pose3d_b_info, "pose3d_b")

    world_w_fp32, world_w_q15 = dump_tensor(interpreter, world_w_info, "world_w")
    world_b_fp32, world_b_q15 = dump_tensor(interpreter, world_b_info, "world_b")

    flag_w_fp32, flag_w_q15 = dump_tensor(interpreter, flag_w_info, "flag_w")
    flag_b_fp32, flag_b_q15 = dump_tensor(interpreter, flag_b_info, "flag_b")

    # -----------------------------------------------------------------
    #  Manifest for convenience
    # -----------------------------------------------------------------
    print("\n[4] Writing simple manifest.json…")
    manifest = {
        "model": MODEL_PATH,
        "weights_dir": OUT_DIR,
        "heads": {
            "pose3d": {
                "w_fp32": "pose3d_w_fp32.npy",
                "w_q15": "pose3d_w_q15.npy",
                "b_fp32": "pose3d_b_fp32.npy",
                "b_q15": "pose3d_b_q15.npy",
                "out_ch": 195,
                "in_ch": 1152,
                "kernel": [2, 2],
            },
            "world": {
                "w_fp32": "world_w_fp32.npy",
                "w_q15": "world_w_q15.npy",
                "b_fp32": "world_b_fp32.npy",
                "b_q15": "world_b_q15.npy",
                "out_ch": 117,
                "in_ch": 1152,
                "kernel": [2, 2],
            },
            "flag": {
                "w_fp32": "flag_w_fp32.npy",
                "w_q15": "flag_w_q15.npy",
                "b_fp32": "flag_b_fp32.npy",
                "b_q15": "flag_b_q15.npy",
                "out_ch": 1,
                "in_ch": 1152,
                "kernel": [2, 2],
            },
        },
    }

    import json
    manifest_path = os.path.join(OUT_DIR, "posehead_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    Manifest written to {manifest_path}")

    print("\n[DONE] Pose-head weights exported.")


if __name__ == "__main__":
    main()
