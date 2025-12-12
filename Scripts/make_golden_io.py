#!/usr/bin/env python3
import os
import numpy as np
import tflite_runtime.interpreter as tflite

# ------------------------------------------------------------------
# Paths / constants
# ------------------------------------------------------------------
MODEL_PATH = "pose_landmark_full.tflite"
INPUT_NPY = "Test_Image/0000.npy"
OUT_DIR = "golden_outputs"

# This is the tensor index we previously identified as the backbone feature
# going into the pose head. If it can't be read (delegate owns it), we skip it.
TENSOR_BACKBONE_FEAT = 335

# ------------------------------------------------------------------
# Helper to load input
# ------------------------------------------------------------------
def load_input():
    arr = np.load(INPUT_NPY)
    # Your .npy is shape (1,1,256,256,3); full model wants (1,256,256,3)
    if arr.ndim == 5 and arr.shape[1] == 1:
        arr = arr[:, 0, ...]
    arr = arr.astype(np.float32, copy=False)
    return arr

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Try to disable XNNPACK (may or may not be honored by this build, but harmless)
    os.environ["TFLITE_DISABLE_XNNPACK"] = "1"

    # 1) Load input
    print("[1] Loading input…")
    inp = load_input()
    print(f"    input shape: {inp.shape} dtype: {inp.dtype}")

    # 2) Run full model
    print("\n[2] Running full TFLite model (delegates disabled where possible)…")

    # Ask for no delegates explicitly
    interpreter = tflite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[]
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("    input_details:", input_details)
    print("    output_details:", output_details)

    # There is only one input
    in_idx = input_details[0]["index"]
    interpreter.set_tensor(in_idx, inp)
    interpreter.invoke()

    # Map the 5 outputs by position:
    #  0: Identity     -> (1,195)   pose3d keypoints
    #  1: Identity_1   -> (1,1)     flag
    #  2: Identity_2   -> (1,256,256,1) segmentation mask
    #  3: Identity_3   -> (1,64,64,39) heatmap/features
    #  4: Identity_4   -> (1,117)   world keypoints
    out0 = interpreter.get_tensor(output_details[0]["index"])  # pose3d
    out1 = interpreter.get_tensor(output_details[1]["index"])  # flag
    out2 = interpreter.get_tensor(output_details[2]["index"])  # seg
    out3 = interpreter.get_tensor(output_details[3]["index"])  # heatmap
    out4 = interpreter.get_tensor(output_details[4]["index"])  # world

    # Save all outputs (FP32)
    np.save(os.path.join(OUT_DIR, "golden_pose3d_fp32.npy"), out0)
    np.save(os.path.join(OUT_DIR, "golden_flag_fp32.npy"),   out1)
    np.save(os.path.join(OUT_DIR, "golden_seg_fp32.npy"),    out2)
    np.save(os.path.join(OUT_DIR, "golden_heatmap_fp32.npy"),out3)
    np.save(os.path.join(OUT_DIR, "golden_world_fp32.npy"),  out4)

    print("    Saved full-model outputs:")
    print("      golden_pose3d_fp32.npy  shape", out0.shape)
    print("      golden_flag_fp32.npy    shape", out1.shape)
    print("      golden_seg_fp32.npy     shape", out2.shape)
    print("      golden_heatmap_fp32.npy shape", out3.shape)
    print("      golden_world_fp32.npy   shape", out4.shape)

    # 3) Try to grab backbone feature tensor
    print("\n[3] Trying to read backbone feature tensor", TENSOR_BACKBONE_FEAT, "…")
    try:
        feat = interpreter.get_tensor(TENSOR_BACKBONE_FEAT)
        np.save(os.path.join(OUT_DIR, "golden_backbone_feat_fp32.npy"), feat)
        print("    SUCCESS: backbone feature saved as golden_backbone_feat_fp32.npy")
        print("             shape:", feat.shape)
    except Exception as e:
        # This is expected if a delegate (like XNNPACK) owns that part of the graph.
        print("    WARNING: could not read backbone feature tensor.")
        print("             Reason:", repr(e))
        print("             This usually means a delegate took over the backbone,")
        print("             so intermediate tensors are not backed by host memory.")
        print("             Final outputs are still valid and saved.")

    # 4) Save the exact FP32 input we used
    np.save(os.path.join(OUT_DIR, "golden_input_fp32.npy"), inp)
    print("\n[4] Saved input as golden_input_fp32.npy")

    print("\n[DONE] Golden IO written under:", OUT_DIR)


if __name__ == "__main__":
    main()
