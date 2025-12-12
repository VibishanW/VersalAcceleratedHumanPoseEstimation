#pragma once
#include <cstdint>

// Common feature size: 2x2x288 = 1152
static constexpr int POSE3D_IN_CH  = 1152;
static constexpr int WORLD_IN_CH   = 1152;
static constexpr int FLAG_IN_CH    = 1152;

// Outputs:
// Pose3D head: 195 outputs (65 keypoints × 3 coords)
static constexpr int POSE3D_OUT_CH = 195;
// World head: 117 outputs
static constexpr int WORLD_OUT_CH  = 117;
// Flag head: 1 output
static constexpr int FLAG_OUT_CH   = 1;

static constexpr int POSE3D_OUT_CH_PAD = POSE3D_OUT_CH + 5; 
static constexpr int WORLD_OUT_CH_PAD  = WORLD_OUT_CH  + 3;
static constexpr int FLAG_OUT_CH_PAD   = FLAG_OUT_CH   + 7; 


// Row-stride (bias + weights) = OUT_CH * (1 + IN_CH)
static constexpr int POSE3D_ROW_STRIDE = 1 + POSE3D_IN_CH;
static constexpr int WORLD_ROW_STRIDE  = 1 + WORLD_IN_CH;
static constexpr int FLAG_ROW_STRIDE   = 1 + FLAG_IN_CH;

// Quantization shifts (tune later if needed)
static constexpr int POSE3D_SHIFT = 15;
static constexpr int WORLD_SHIFT  = 15;
static constexpr int FLAG_SHIFT   = 15;

// Simple saturate helpers (if you need them on host for debugging)
static inline int16_t sat_q15(int32_t x)
{
    if (x >  32767) return  32767;
    if (x < -32768) return -32768;
    return static_cast<int16_t>(x);
}
