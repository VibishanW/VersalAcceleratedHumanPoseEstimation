#pragma once
#include <adf.h>
#include <cstdint>

#include "pose_head_params.hpp"
#include "pose3d_head_kernel.hpp"
#include "world_head_kernel.hpp"
#include "flag_head_kernel.hpp"

using namespace adf;

// Top-level BlazePose pose-head graph: pose3D, world, flag (GMIO-based)
class PoseHeadGraph : public graph {
public:
    // GMIO feature inputs (all share same backbone feature tensor)
    input_gmio  pose3d_feat_gmio;
    input_gmio  world_feat_gmio;
    input_gmio  flag_feat_gmio;

    // GMIO weight inputs
    input_gmio  pose3d_w_gmio;
    input_gmio  world_w_gmio;
    input_gmio  flag_w_gmio;

    // GMIO outputs
    output_gmio pose3d_out_gmio;
    output_gmio world_out_gmio;
    output_gmio flag_out_gmio;

    // Kernels
    kernel pose3d_k;
    kernel world_k;
    kernel flag_k;

    PoseHeadGraph() {
        // 128-bit GMIO width, depth=1000 is plenty for our bursts
        pose3d_feat_gmio = input_gmio::create("pose3d_feat_gmio", 128, 1000);
        world_feat_gmio  = input_gmio::create("world_feat_gmio",  128, 1000);
        flag_feat_gmio   = input_gmio::create("flag_feat_gmio",   128, 1000);

        pose3d_w_gmio    = input_gmio::create("pose3d_w_gmio",    128, 1000);
        world_w_gmio     = input_gmio::create("world_w_gmio",     128, 1000);
        flag_w_gmio      = input_gmio::create("flag_w_gmio",      128, 1000);

        pose3d_out_gmio  = output_gmio::create("pose3d_out_gmio", 128, 1000);
        world_out_gmio   = output_gmio::create("world_out_gmio",  128, 1000);
        flag_out_gmio    = output_gmio::create("flag_out_gmio",   128, 1000);

        // Create kernels
        pose3d_k = kernel::create(pose3d_head_kernel);
        world_k  = kernel::create(world_head_kernel);
        flag_k   = kernel::create(flag_head_kernel);

        source(pose3d_k) = "pose3d_head_kernel.cpp";
        source(world_k)  = "world_head_kernel.cpp";
        source(flag_k)   = "flag_head_kernel.cpp";

        runtime<ratio>(pose3d_k) = 1.0;
        runtime<ratio>(world_k)  = 1.0;
        runtime<ratio>(flag_k)   = 1.0;

        // Features: GMIO -> window
        connect< window<POSE3D_IN_CH * sizeof(int16_t)> >(
            pose3d_feat_gmio.out[0], pose3d_k.in[0]);

        connect< window<WORLD_IN_CH * sizeof(int16_t)> >(
            world_feat_gmio.out[0], world_k.in[0]);

        connect< window<FLAG_IN_CH * sizeof(int16_t)> >(
            flag_feat_gmio.out[0], flag_k.in[0]);

        // Weights: GMIO -> stream
        connect<>(pose3d_w_gmio.out[0], pose3d_k.in[1]);
        connect<>(world_w_gmio.out[0],  world_k.in[1]);
        connect<>(flag_w_gmio.out[0],   flag_k.in[1]);

        // Outputs: stream -> GMIO
        connect<>(pose3d_k.out[0], pose3d_out_gmio.in[0]);
        connect<>(world_k.out[0],  world_out_gmio.in[0]);
        connect<>(flag_k.out[0],   flag_out_gmio.in[0]);
    }
};

// Global instance used by aiecompiler and host
extern PoseHeadGraph g;

// Simple PS / host sim helper
static inline void ps_main() {
    g.init();
    g.run(1);
    g.end();
}
