#pragma once

#include <ap_int.h>
#include <hls_stream.h>

// Memory -> stream
void feat_mover(ap_uint<128>* mem,
                hls::stream<ap_uint<128>>& outStream,
                unsigned int num_words,
                unsigned int start_addr);

// Memory -> stream (weights)
void weight_mover(ap_uint<128>* mem,
                  hls::stream<ap_uint<128>>& outStream,
                  unsigned int num_words,
                  unsigned int start_addr);

// Stream -> memory
void output_mover(hls::stream<ap_uint<128>>& inStream,
                  ap_uint<128>* mem,
                  unsigned int num_words,
                  unsigned int start_addr);
