# Makefile for Custom_Pl_Kernels
# - Testbenches to verify movers
# - Vitis kernel compilation (.xo) for integration with overlay

# Vitis HLS include directory (for ap_int.h, hls_stream.h, etc.)
HLS_INC ?= /share/Xilinx/Vitis_HLS/2023.2/include
VPP      ?= v++
PLATFORM ?= /opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm
TARGET   ?= hw

# Host C++ compiler flags
CXX      := g++
CXXFLAGS := -std=c++14 -I$(HLS_INC) -Wall -Wextra

# Testbench executables
TB_EXES  := tb_feat_mover tb_weight_mover tb_output_mover

# Kernel names and corresponding .xo objects
KERNELS   := feat_mover weight_mover output_mover
XO_OBJS   := $(KERNELS:%=%.xo)

# Common v++ flags for PL kernels
XOCCFLAGS := --platform $(PLATFORM) -t $(TARGET) -s -g

# ---------------------------------------------------------------------
# Default target: build testbenches AND kernel .xo files
# ---------------------------------------------------------------------
all: $(TB_EXES) xo

# ---------------------------------------------------------------------
# Testbenches
# ---------------------------------------------------------------------
tb_feat_mover: feat_mover.cpp tb_feat_mover.cpp movers.hpp
	$(CXX) $(CXXFLAGS) $^ -o $@

tb_weight_mover: weight_mover.cpp tb_weight_mover.cpp movers.hpp
	$(CXX) $(CXXFLAGS) $^ -o $@

tb_output_mover: output_mover.cpp tb_output_mover.cpp movers.hpp
	$(CXX) $(CXXFLAGS) $^ -o $@

run: all
	./tb_feat_mover
	./tb_weight_mover
	./tb_output_mover

# ---------------------------------------------------------------------
# Compile data mover kernels to .xo
# ---------------------------------------------------------------------
compile: xo

xo: $(XO_OBJS)

%.xo: %.cpp movers.hpp
	$(VPP) $(XOCCFLAGS) --kernel $* -c -o $@ $<

# ---------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------
clean:
	rm -f $(TB_EXES) $(XO_OBJS) *.log
	rm -rf _x

.PHONY: all run xo compile clean