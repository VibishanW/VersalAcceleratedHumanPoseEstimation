#include "world_head_kernel.hpp"

using namespace adf;

// DM-resident feature buffer for world head
alignas(32) static int16_t world_feat_buf[WORLD_IN_CH];

void world_head_kernel(input_window<int16>*  feat_win,
                       input_stream<int16>*  w_stream,
                       output_stream<int16>* out_stream)
{
    // Load features from the input window into DM buffer
    for (int i = 0; i < WORLD_IN_CH; ++i) {
        world_feat_buf[i] = window_readincr(feat_win);
    }

    // Output channels
    for (int oc = 0; oc < WORLD_OUT_CH; ++oc) {

        // Bias term
        int32_t acc = static_cast<int32_t>(readincr(w_stream));

        // MAC over all input channels
        for (int ic = 0; ic < WORLD_IN_CH; ++ic) {
            int16_t f = world_feat_buf[ic];
            int16_t w = readincr(w_stream);
            acc += static_cast<int32_t>(f) * static_cast<int32_t>(w);
        }

        // shift + saturate to Q15
        acc >>= WORLD_SHIFT;
        writeincr(out_stream, sat_q15(acc));
    }

    // Emit padded zeros WITHOUT touching the weight stream
    for (int oc = WORLD_OUT_CH; oc < WORLD_OUT_CH_PAD; ++oc) {
        writeincr(out_stream, 0);
    }
}
