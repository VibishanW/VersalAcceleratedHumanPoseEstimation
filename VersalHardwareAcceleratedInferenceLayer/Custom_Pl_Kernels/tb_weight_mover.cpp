#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include "movers.hpp"

int main() {
    const unsigned MEM_SIZE = 64;
    const unsigned N        = 16;
    const unsigned START    = 2;

    ap_uint<128> mem[MEM_SIZE];
    hls::stream<ap_uint<128>> s;

    // Initialize memory with known pattern
    for (unsigned i = 0; i < MEM_SIZE; ++i)
        mem[i] = 1000 + i;

    // Call kernel
    weight_mover(mem, s, N, START);

    bool ok = true;
    for (unsigned i = 0; i < N; ++i) {
        if (s.empty()) {
            std::cout << "ERROR: Stream underflow at i=" << i << "\n";
            ok = false;
            break;
        }
        ap_uint<128> v = s.read();
        ap_uint<128> exp = 1000 + START + i;
        if (v != exp) {
            std::cout << "Mismatch at i=" << i
                      << " got=" << (unsigned long long)v
                      << " expected=" << (unsigned long long)exp << "\n";
            ok = false;
        }
    }

    if (ok)
        std::cout << "weight_mover PASS\n";
    else
        std::cout << "weight_mover FAIL\n";

    return ok ? 0 : 1;
}
