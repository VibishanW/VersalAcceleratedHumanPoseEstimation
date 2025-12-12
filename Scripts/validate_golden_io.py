#!/usr/bin/env python3
import os
import numpy as np
import tflite_runtime.interpreter as tflite

# -------------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------------
MODEL_PATH = "pose_landmark_full.tflite"
GOLDEN_DIR = "golden_outputs"

GOLDEN_INPUT = os.path.join(GOLDEN_DIR, "golden_input_fp32.npy")
GOLDEN_POSE3D = os.path.join(GOLDEN_DIR, "golden_pose3d_fp32.npy")
GOLDEN_FLAG   = os.path.join(GOLDEN_DIR, "golden_flag_fp32.npy")
GOLDEN_SEG    = os.path.join(GOLDEN_DIR, "golden_seg_fp32.npy")
GOLDEN_HM     = os.path.join(GOLDEN_DIR, "golden_heatmap_fp32.npy")
GOLDEN_WORLD  = os.path.join(GOLDEN_DIR, "golden_world_fp32.npy")

# Comparison tolerances (these are tight for FP32, but reasonable)
RTOL = 1e-5
ATOL = 1e-7

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def run_model(input_array):
    """Run the TFLite model on input_array and return the 5 outputs."""
    # Try to keep delegates off (if runtime honors it)
    os.environ["TFLITE_DISABLE_XNNPACK"] = "1"

    intr = tflite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[]
    )
    intr.allocate_tensors()

    input_details = intr.get_input_details()
    output_details = intr.get_output_details()

    print("[run_model] input_details:", input_details)
    print("[run_model] output_details:", output_details)

    # Assume single input tensor
    in_idx = input_details[0]["index"]
    intr.set_tensor(in_idx, input_array)
    intr.invoke()

    # Same order as we used in make_golden_io.py
    out_pose3d = intr.get_tensor(output_details[0]["index"])  # (1,195)
    out_flag   = intr.get_tensor(output_details[1]["index"])  # (1,1)
    out_seg    = intr.get_tensor(output_details[2]["index"])  # (1,256,256,1)
    out_hm     = intr.get_tensor(output_details[3]["index"])  # (1,64,64,39)
    out_world  = intr.get_tensor(output_details[4]["index"])  # (1,117)

    return out_pose3d, out_flag, out_seg, out_hm, out_world


def compare_arrays(name, golden, current):
    """Compare two arrays and print detailed error stats."""
    print(f"\n=== {name} ===")

    if golden.shape != current.shape:
        print(f"  SHAPE MISMATCH!")
        print(f"    golden: {golden.shape}")
        print(f"    current: {current.shape}")
        return False

    diff = current - golden
    abs_diff = np.abs(diff)

    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    rms = float(np.sqrt(np.mean(diff ** 2)))
    denom = np.maximum(np.abs(golden), 1e-8)
    max_rel = float((abs_diff / denom).max())

    allclose = np.allclose(current, golden, rtol=RTOL, atol=ATOL)

    print(f"  shape        : {golden.shape}")
    print(f"  max_abs_err  : {max_abs:.6e}")
    print(f"  mean_abs_err : {mean_abs:.6e}")
    print(f"  rms_err      : {rms:.6e}")
    print(f"  max_rel_err  : {max_rel:.6e}")
    print(f"  allclose(rtol={RTOL}, atol={ATOL}): {allclose}")

    return allclose


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("[1] Loading golden input/output files…")

    if not os.path.exists(GOLDEN_INPUT):
        raise FileNotFoundError(f"Missing golden input: {GOLDEN_INPUT}")

    golden_input = np.load(GOLDEN_INPUT)
    print(f"    golden_input shape: {golden_input.shape}, dtype: {golden_input.dtype}")

    golden_pose3d = np.load(GOLDEN_POSE3D)
    golden_flag   = np.load(GOLDEN_FLAG)
    golden_seg    = np.load(GOLDEN_SEG)
    golden_hm     = np.load(GOLDEN_HM)
    golden_world  = np.load(GOLDEN_WORLD)

    print("    Loaded golden outputs:")
    print(f"      pose3d : {golden_pose3d.shape}")
    print(f"      flag   : {golden_flag.shape}")
    print(f"      seg    : {golden_seg.shape}")
    print(f"      heatmap: {golden_hm.shape}")
    print(f"      world  : {golden_world.shape}")

    print("\n[2] Re-running TFLite model on golden input…")
    cur_pose3d, cur_flag, cur_seg, cur_hm, cur_world = run_model(golden_input)

    print("\n[3] Comparing re-run outputs against golden_*_fp32.npy …")

    ok_pose3d = compare_arrays("POSE3D", golden_pose3d, cur_pose3d)
    ok_flag   = compare_arrays("FLAG",   golden_flag,   cur_flag)
    ok_seg    = compare_arrays("SEG",    golden_seg,    cur_seg)
    ok_hm     = compare_arrays("HEATMAP",golden_hm,     cur_hm)
    ok_world  = compare_arrays("WORLD",  golden_world,  cur_world)

    all_ok = ok_pose3d and ok_flag and ok_seg and ok_hm and ok_world

    print("\n[4] SUMMARY")
    print(f"  POSE3D : {'OK' if ok_pose3d else 'MISMATCH'}")
    print(f"  FLAG   : {'OK' if ok_flag   else 'MISMATCH'}")
    print(f"  SEG    : {'OK' if ok_seg    else 'MISMATCH'}")
    print(f"  HEATMAP: {'OK' if ok_hm     else 'MISMATCH'}")
    print(f"  WORLD  : {'OK' if ok_world  else 'MISMATCH'}")
    print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
