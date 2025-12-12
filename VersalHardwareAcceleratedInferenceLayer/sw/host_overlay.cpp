// host.cpp — GMIO-based BlazePose 3-head test (pose3D + world + flag)

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_graph.h"
#include "experimental/xrt_aie.h"

#include "../aie_overlay/Kernels/pose_head_params.hpp"

//---------------------------------------------------------
// Helpers
//---------------------------------------------------------
template <typename T>
void load_bin(const std::string &path, T *buf, std::size_t elems)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
        throw std::runtime_error("Failed to open binary file: " + path);

    fin.seekg(0, std::ios::end);
    std::streamsize fsize = fin.tellg();
    fin.seekg(0, std::ios::beg);

    const std::streamsize needed =
        static_cast<std::streamsize>(elems * sizeof(T));
    if (fsize < needed) {
        throw std::runtime_error("File too small: " + path +
                                 " (have " + std::to_string(fsize) +
                                 " bytes, need " + std::to_string(needed) + ")");
    }

    fin.read(reinterpret_cast<char *>(buf), needed);
    if (!fin)
        throw std::runtime_error("Failed to read all data from: " + path);
}

template <typename T>
void save_txt(const std::string &path, const T *buf, std::size_t elems)
{
    std::ofstream fout(path);
    if (!fout)
        throw std::runtime_error("Failed to open output file: " + path);

    for (std::size_t i = 0; i < elems; ++i)
        fout << static_cast<long>(buf[i]) << "\n";
}

//---------------------------------------------------------
// MAIN
//---------------------------------------------------------
int main(int argc, char *argv[])
{
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0]
                      << " <pose_heads_hw[_emu].xclbin> <data_dir>\n";
            return 1;
        }

        const std::string xclbin_path = argv[1];
        const std::string data_dir    = argv[2];

        std::cout << "Opening device 0...\n";
        xrt::device device{0};

        std::cout << "Loading xclbin: " << xclbin_path << "\n";
        auto uuid = device.load_xclbin(xclbin_path);

        std::cout << "Opening graph: g\n";
        xrt::graph graph(device, uuid, "g");

        //-------------------------------------------------
        // Sizes per head
        //-------------------------------------------------
        // Inputs
        const std::size_t pose3d_feat_elems =
            static_cast<std::size_t>(POSE3D_IN_CH);
        const std::size_t world_feat_elems =
            static_cast<std::size_t>(WORLD_IN_CH);
        const std::size_t flag_feat_elems  =
            static_cast<std::size_t>(FLAG_IN_CH);

        // Weights (rows * stride)
        const std::size_t pose3d_w_elems =
            static_cast<std::size_t>(POSE3D_OUT_CH) *
            static_cast<std::size_t>(POSE3D_ROW_STRIDE);

        const std::size_t world_w_elems =
            static_cast<std::size_t>(WORLD_OUT_CH) *
            static_cast<std::size_t>(WORLD_ROW_STRIDE);

        const std::size_t flag_w_elems  =
            static_cast<std::size_t>(FLAG_OUT_CH) *
            static_cast<std::size_t>(FLAG_ROW_STRIDE);

        // Outputs
        const std::size_t pose3d_out_elems =
            static_cast<std::size_t>(POSE3D_OUT_CH);
        const std::size_t world_out_elems  =
            static_cast<std::size_t>(WORLD_OUT_CH);
        const std::size_t flag_out_elems   =
            static_cast<std::size_t>(FLAG_OUT_CH);

        using xrt::aie::bo;

        //-------------------------------------------------
        // Allocate BOs (DDR buffers used by GMIO)
        //-------------------------------------------------
        // Features (one per head, but same data)
        bo pose3d_feat_bo(
            device,
            pose3d_feat_elems * sizeof(int16_t),
            xrt::bo::flags::normal,
            /*bank=*/0);

        bo world_feat_bo(
            device,
            world_feat_elems * sizeof(int16_t),
            xrt::bo::flags::normal,
            /*bank=*/0);

        bo flag_feat_bo(
            device,
            flag_feat_elems * sizeof(int16_t),
            xrt::bo::flags::normal,
            /*bank=*/0);

        // Weights
        bo pose3d_w_bo(
            device,
            pose3d_w_elems * sizeof(int16_t),
            xrt::bo::flags::normal,
            /*bank=*/0);

        bo world_w_bo(
            device,
            world_w_elems * sizeof(int16_t),
            xrt::bo::flags::normal,
            /*bank=*/0);

        bo flag_w_bo(
            device,
            flag_w_elems * sizeof(int16_t),
            xrt::bo::flags::normal,
            /*bank=*/0);

        // Outputs
        bo pose3d_out_bo(
            device,
            pose3d_out_elems * sizeof(int16_t),
            xrt::bo::flags::normal,
            /*bank=*/0);

        bo world_out_bo(
            device,
            world_out_elems * sizeof(int16_t),
            xrt::bo::flags::normal,
            /*bank=*/0);

        bo flag_out_bo(
            device,
            flag_out_elems * sizeof(int16_t),
            xrt::bo::flags::normal,
            /*bank=*/0);

        //-------------------------------------------------
        // Map to host pointers
        //-------------------------------------------------
        auto pose3d_feat = pose3d_feat_bo.map<int16_t *>();
        auto world_feat  = world_feat_bo.map<int16_t *>();
        auto flag_feat   = flag_feat_bo.map<int16_t *>();

        auto pose3d_w = pose3d_w_bo.map<int16_t *>();
        auto world_w  = world_w_bo.map<int16_t *>();
        auto flag_w   = flag_w_bo.map<int16_t *>();

        auto pose3d_out = pose3d_out_bo.map<int16_t *>();
        auto world_out  = world_out_bo.map<int16_t *>();
        auto flag_out   = flag_out_bo.map<int16_t *>();

        //-------------------------------------------------
        // Load feature + weight binaries
        //-------------------------------------------------
        const std::string feat_bin      = data_dir + "/posehead_input_q15.bin";
        const std::string pose3d_w_bin  = data_dir + "/pose3d_fc_q15.bin";
        const std::string world_w_bin   = data_dir + "/world_fc_q15.bin";
        const std::string flag_w_bin    = data_dir + "/flag_fc_q15.bin";

        std::cout << "Loading feature + weight binaries from: " << data_dir << "\n";

        // Same features for all 3 heads (loaded three times into 3 BOs)
        load_bin<int16_t>(feat_bin, pose3d_feat, pose3d_feat_elems);
        load_bin<int16_t>(feat_bin, world_feat,  world_feat_elems);
        load_bin<int16_t>(feat_bin, flag_feat,   flag_feat_elems);

        // Per-head weight matrices
        load_bin<int16_t>(pose3d_w_bin, pose3d_w, pose3d_w_elems);
        load_bin<int16_t>(world_w_bin,  world_w,  world_w_elems);
        load_bin<int16_t>(flag_w_bin,   flag_w,   flag_w_elems);

        //-------------------------------------------------
        // Start GMIO transfers
        //-------------------------------------------------
        std::cout << "Starting GMIO transfers and graph run...\n";

        // Features -> AIE
        pose3d_feat_bo.async(
            "g.pose3d_feat_gmio",
            XCL_BO_SYNC_BO_GMIO_TO_AIE,
            pose3d_feat_elems * sizeof(int16_t),
            /*offset=*/0);

        world_feat_bo.async(
            "g.world_feat_gmio",
            XCL_BO_SYNC_BO_GMIO_TO_AIE,
            world_feat_elems * sizeof(int16_t),
            /*offset=*/0);

        flag_feat_bo.async(
            "g.flag_feat_gmio",
            XCL_BO_SYNC_BO_GMIO_TO_AIE,
            flag_feat_elems * sizeof(int16_t),
            /*offset=*/0);

        // Weights -> AIE
        pose3d_w_bo.async(
            "g.pose3d_w_gmio",
            XCL_BO_SYNC_BO_GMIO_TO_AIE,
            pose3d_w_elems * sizeof(int16_t),
            /*offset=*/0);

        world_w_bo.async(
            "g.world_w_gmio",
            XCL_BO_SYNC_BO_GMIO_TO_AIE,
            world_w_elems * sizeof(int16_t),
            /*offset=*/0);

        flag_w_bo.async(
            "g.flag_w_gmio",
            XCL_BO_SYNC_BO_GMIO_TO_AIE,
            flag_w_elems * sizeof(int16_t),
            /*offset=*/0);

        // Outputs: AIE -> DDR
        pose3d_out_bo.async(
            "g.pose3d_out_gmio",
            XCL_BO_SYNC_BO_AIE_TO_GMIO,
            pose3d_out_elems * sizeof(int16_t),
            /*offset=*/0);

        world_out_bo.async(
            "g.world_out_gmio",
            XCL_BO_SYNC_BO_AIE_TO_GMIO,
            world_out_elems * sizeof(int16_t),
            /*offset=*/0);

        flag_out_bo.async(
            "g.flag_out_gmio",
            XCL_BO_SYNC_BO_AIE_TO_GMIO,
            flag_out_elems * sizeof(int16_t),
            /*offset=*/0);

        //-------------------------------------------------
        // Run the graph
        //-------------------------------------------------
        std::cout << "Running graph for 1 iteration...\n";
        graph.run(1);
        graph.wait();
        std::cout << "Graph completed.\n";

        //-------------------------------------------------
        // Sync DDR -> host and save outputs
        //-------------------------------------------------
        pose3d_out_bo.sync(
            "g.pose3d_out_gmio",
            XCL_BO_SYNC_BO_FROM_DEVICE);

        world_out_bo.sync(
            "g.world_out_gmio",
            XCL_BO_SYNC_BO_FROM_DEVICE);

        flag_out_bo.sync(
            "g.flag_out_gmio",
            XCL_BO_SYNC_BO_FROM_DEVICE);

        std::cout << "Saving outputs into data_dir...\n";
        save_txt<int16_t>(data_dir + "/pose3d_out_hw.txt",
                          pose3d_out, pose3d_out_elems);
        save_txt<int16_t>(data_dir + "/world_out_hw.txt",
                          world_out, world_out_elems);
        save_txt<int16_t>(data_dir + "/flag_out_hw.txt",
                          flag_out, flag_out_elems);

        std::cout << "Done.\n";
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}