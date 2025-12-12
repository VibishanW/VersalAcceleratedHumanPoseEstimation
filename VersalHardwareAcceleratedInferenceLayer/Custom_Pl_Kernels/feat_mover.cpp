#include "movers.hpp"

// Simple memory-to-stream mover for feature tensors.
// Reads num_words 128-bit words from mem[start_addr + i] and writes to outStream.
void feat_mover(ap_uint<128>* mem,
                hls::stream<ap_uint<128>>& outStream,
                unsigned int num_words,
                unsigned int start_addr) {
#pragma HLS INTERFACE m_axi     port=mem        offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=mem                        bundle=control
#pragma HLS INTERFACE axis      port=outStream
#pragma HLS INTERFACE s_axilite port=num_words                  bundle=control
#pragma HLS INTERFACE s_axilite port=start_addr                 bundle=control
#pragma HLS INTERFACE s_axilite port=return                     bundle=control

READ_LOOP:
    for (unsigned int i = 0; i < num_words; ++i) {
#pragma HLS PIPELINE II=1
        ap_uint<128> v = mem[start_addr + i];
        outStream.write(v);
    }
}
