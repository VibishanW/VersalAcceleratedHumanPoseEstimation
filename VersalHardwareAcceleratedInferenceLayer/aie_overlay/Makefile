# Copyright 2021 Xilinx Inc.
#
# AIE-only build for BlazePose pose-head graph.

.PHONY: all clean aie_compile aie_simulate aie_compile_x86 aie_simulate_x86

all: aie_compile aie_simulate
all_x86: aie_compile_x86 aie_simulate_x86

PLATFORM := /opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm

clean:
	-@rm -rf .Xil .ipcache vivado* *.xpe *.txt *.log
	-@rm -rf Work libadf.a
	-@rm -rf x86simulator_output aiesimulator_output xnwOut .AIE_SIM_CMD_LINE_OPTIONS \
	        pl_sample_count* *.html ISS_RPC_SERVER_PORT

# ---------------------------------------------------------
# AIE compile (hardware)
# ---------------------------------------------------------
aie_compile: libadf.a

libadf.a: Graph/graph.cpp Kernels/*.cpp
	@echo "INFO: Running aiecompiler for hw..."
	@rm -rf Work libadf.a
	@mkdir -p Work
	@aiecompiler \
	  --platform=$(PLATFORM) \
	  --target=hw \
	  --include=Graph \
	  --include=Kernels \
	  --include=Data \
	  --workdir=./Work \
	  Graph/graph.cpp Kernels/*.cpp

# ---------------------------------------------------------
# AIE compile (x86)
# ---------------------------------------------------------
aie_compile_x86:
	@echo "INFO: Running aiecompiler for x86sim..."
	@rm -rf Work libadf.a
	@mkdir -p Work
	@aiecompiler \
	  --platform=$(PLATFORM) \
	  --target=x86sim \
	  --include=Graph \
	  --include=Kernels \
	  --include=Data \
	  --workdir=./Work \
	  Graph/graph.cpp Kernels/*.cpp

# ---------------------------------------------------------
# AIE simulation
# ---------------------------------------------------------
aie_simulate:
	@echo "INFO: Running aiesimulator..."
	@aiesimulator --pkg-dir=./Work --profile

aie_simulate_x86:
	@echo "INFO: Running x86simulator..."
	@x86simulator --pkg-dir=./Work