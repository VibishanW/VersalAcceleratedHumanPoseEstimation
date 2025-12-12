#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include "movers.hpp"

int main() {
    const unsigned MEM_SIZE = 64;
    const unsigned N        = 16;
    const unsigned START    = 4;

    ap_uint<128> mem[MEM_SIZE];
    hls::stream<ap_uint<128>> s;

    // Initialize memory with known pattern
    for (unsigned i = 0; i < MEM_SIZE; ++i)
        mem[i] = i;

    // Call kernel
    feat_mover(mem, s, N, START);

    bool ok = true;
    for (unsigned i = 0; i < N; ++i) {
        if (s.empty()) {
            std::cout << "ERROR: Stream underflow at i=" << i << "\n";
            ok = false;
            break;
        }
        ap_uint<128> v = s.read();
        ap_uint<128> exp = START + i;
        if (v != exp) {
            std::cout << "Mismatch at i=" << i
                      << " got=" << (unsigned long long)v
                      << " expected=" << (unsigned long long)exp << "\n";
            ok = false;
        }
    }

    if (ok)
        std::cout << "feat_mover PASS\n";
    else
        std::cout << "feat_mover FAIL\n";

    return ok ? 0 : 1;
}
