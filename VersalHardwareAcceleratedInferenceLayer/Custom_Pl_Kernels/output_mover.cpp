#include "movers.hpp"

// Simple stream-to-memory mover for output tensors.
// Reads num_words words from inStream and writes to mem[start_addr + i].
void output_mover(hls::stream<ap_uint<128>>& inStream,
                  ap_uint<128>* mem,
                  unsigned int num_words,
                  unsigned int start_addr) {
#pragma HLS INTERFACE axis      port=inStream
#pragma HLS INTERFACE m_axi     port=mem        offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=mem                        bundle=control
#pragma HLS INTERFACE s_axilite port=num_words                  bundle=control
#pragma HLS INTERFACE s_axilite port=start_addr                 bundle=control
#pragma HLS INTERFACE s_axilite port=return                     bundle=control

WRITE_LOOP:
    for (unsigned int i = 0; i < num_words; ++i) {
#pragma HLS PIPELINE II=1
        ap_uint<128> v = inStream.read();
        mem[start_addr + i] = v;
    }
}
