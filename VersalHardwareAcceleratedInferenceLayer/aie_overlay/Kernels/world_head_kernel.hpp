#pragma once
#include <adf.h>
#include <cstdint>
#include "pose_head_params.hpp"

using namespace adf;

void world_head_kernel(input_window<int16>*  feat_win,
                       input_stream<int16>*  w_stream,
                       output_stream<int16>* out_stream);
