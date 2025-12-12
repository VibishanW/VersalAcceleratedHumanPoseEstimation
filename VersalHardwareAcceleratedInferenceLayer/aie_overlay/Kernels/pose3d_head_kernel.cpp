#include "pose3d_head_kernel.hpp"

using namespace adf;

// DM-resident buffer for the input feature vector
alignas(32) static int16_t pose3d_feat_buf[POSE3D_IN_CH];

void pose3d_head_kernel(input_window<int16>*  feat_win,
                        input_stream<int16>*  w_stream,
                        output_stream<int16>* out_stream)
{
    // ------------------------------------------------------------------
    // 1. Load feature vector from window
    //    Expect exactly POSE3D_IN_CH = 1152 int16 samples
    // ------------------------------------------------------------------
    for (int i = 0; i < POSE3D_IN_CH; ++i) {
        pose3d_feat_buf[i] = window_readincr(feat_win);
    }

    // ------------------------------------------------------------------
    // 2. Fully-connected layer:
    //    For each output channel oc:
    //      - Read bias from w_stream
    //      - Read POSE3D_IN_CH weights from w_stream
    //      - MAC: acc = bias + sum(f[ic] * w[ic])
    //      - Shift + saturate to Q15
    //
    //    This exactly matches pose3d_w.txt layout:
    //      For oc in [0..POSE3D_OUT_CH-1]:
    //          bias_oc,
    //          w_oc[0], w_oc[1], ..., w_oc[POSE3D_IN_CH-1]
    // ------------------------------------------------------------------
    for (int oc = 0; oc < POSE3D_OUT_CH; ++oc) {

        // Bias term (Q15)
        int32_t acc = static_cast<int32_t>(readincr(w_stream));

        // MAC over all input channels
        for (int ic = 0; ic < POSE3D_IN_CH; ++ic) {
            int16_t f = pose3d_feat_buf[ic];
            int16_t w = readincr(w_stream);
            acc += static_cast<int32_t>(f) * static_cast<int32_t>(w);
        }

        // Shift + saturate to Q15
        acc >>= POSE3D_SHIFT;
        writeincr(out_stream, sat_q15(acc));
    }

    // ------------------------------------------------------------------
    // 3. Emit padded zeros up to POSE3D_OUT_CH_PAD
    //    (keeps the same padded output length you used before)
    // ------------------------------------------------------------------
    for (int oc = POSE3D_OUT_CH; oc < POSE3D_OUT_CH_PAD; ++oc) {
        writeincr(out_stream, 0);
    }
}
