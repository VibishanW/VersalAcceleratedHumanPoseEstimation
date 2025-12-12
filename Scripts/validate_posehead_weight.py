#!/usr/bin/env python3
import os
import json
import numpy as np

os.environ.setdefault("TFLITE_DISABLE_XNNPACK", "1")

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

MODEL_PATH = "pose_landmark_full.tflite"
WEIGHTS_DIR = "weights"


def find_tensor(details, name_substr, expected_shape):
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


def compare(name, arr_model, arr_file):
    arr_model = arr_model.astype(np.float32)
    arr_file = arr_file.astype(np.float32)

    if arr_model.shape != arr_file.shape:
        print(f"  {name}: SHAPE MISMATCH model={arr_model.shape}, file={arr_file.shape}")
        return

    diff = arr_model - arr_file
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    rms = np.sqrt(np.mean(diff ** 2))
    max_rel = np.max(np.abs(diff) / (np.abs(arr_model) + 1e-12))

    print(f"  {name}:")
    print(f"    shape        : {arr_model.shape}")
    print(f"    max_abs_err  : {max_abs: .6e}")
    print(f"    mean_abs_err : {mean_abs: .6e}")
    print(f"    rms_err      : {rms: .6e}")
    print(f"    max_rel_err  : {max_rel: .6e}")


def main():
    print("[1] Loading manifest and TFLite model…")
    manifest_path = os.path.join(WEIGHTS_DIR, "posehead_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    print("    allocate_tensors done, #tensors =", len(details))

    # Pose3D
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

    # World
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

    # Flag
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

    print("\n[2] Comparing tensors against exported FP32 .npy files…")

    heads = ["pose3d", "world", "flag"]
    tensor_infos = {
        "pose3d": (pose3d_w_info, pose3d_b_info),
        "world": (world_w_info, world_b_info),
        "flag": (flag_w_info, flag_b_info),
    }

    for head in heads:
        print(f"\n=== {head.upper()} ===")
        head_manifest = manifest["heads"][head]

        # Weights
        w_file = os.path.join(WEIGHTS_DIR, head_manifest["w_fp32"])
        b_file = os.path.join(WEIGHTS_DIR, head_manifest["b_fp32"])

        w_saved = np.load(w_file)
        b_saved = np.load(b_file)

        w_model = interpreter.get_tensor(tensor_infos[head][0]["index"])
        b_model = interpreter.get_tensor(tensor_infos[head][1]["index"])

        compare("weights", w_model, w_saved)
        compare("bias", b_model, b_saved)

    print("\n[DONE] Weight validation finished.")


if __name__ == "__main__":
    main()
