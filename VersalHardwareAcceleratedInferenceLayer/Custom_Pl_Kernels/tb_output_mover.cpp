#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include "movers.hpp"

int main() {
    const unsigned MEM_SIZE = 64;
    const unsigned N        = 16;
    const unsigned START    = 8;

    ap_uint<128> mem[MEM_SIZE];
    hls::stream<ap_uint<128>> s;

    // Clear memory
    for (unsigned i = 0; i < MEM_SIZE; ++i)
        mem[i] = 0;

    // Push known pattern into stream
    for (unsigned i = 0; i < N; ++i)
        s.write(2000 + i);

    // Call kernel
    output_mover(s, mem, N, START);

    bool ok = true;
    for (unsigned i = 0; i < N; ++i) {
        ap_uint<128> v   = mem[START + i];
        ap_uint<128> exp = 2000 + i;
        if (v != exp) {
            std::cout << "Mismatch at mem[" << (START + i)
                      << "] got=" << (unsigned long long)v
                      << " expected=" << (unsigned long long)exp << "\n";
            ok = false;
        }
    }

    if (ok)
        std::cout << "output_mover PASS\n";
    else
        std::cout << "output_mover FAIL\n";

    return ok ? 0 : 1;
}
