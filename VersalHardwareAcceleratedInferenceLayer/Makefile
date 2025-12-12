# Top-level Makefile for BlazePose pose-head design for VCK5000

ECHO=@echo

.PHONY: help build_hw build_sw all \
        aie custom_pl hw sw \
        clean clean_hw clean_sw clean_all \
        AIELink forMetrics

# Global config (Change path for your local cluster/setup)
PLATFORM_REPO_PATHS ?= /opt/xilinx/platforms
PLATFORM ?= /opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm  

# Build target: hw or hw_emu
TARGET := hw

export PLATFORM_REPO_PATHS
export PLATFORM
export TARGET

# --------------------------------------------------------------------
# Help
# --------------------------------------------------------------------

help:
	$(ECHO) ""
	$(ECHO) "Top-level Makefile Usage:"
	$(ECHO) "  make build_hw            # Build AIE + PL + XCLBIN for hardware"
	$(ECHO) "  make build_hw TARGET=hw_emu"
	$(ECHO) "                          # Same, but for hardware emulation"
	$(ECHO) ""
	$(ECHO) "  make build_sw            # Build host executable in ./sw"
	$(ECHO) ""
	$(ECHO) "  make all                 # build_hw then build_sw"
	$(ECHO) ""
	$(ECHO) "  make AIELink             # AIE + PL link-only metrics (util/power)"
	$(ECHO) "  make forMetrics          # Minimal AIE-only hw_emu xclbin for metrics"
	$(ECHO) ""
	$(ECHO) "  make clean_all           # Clean all sub-projects"
	$(ECHO) ""

# Default: do both
all: build_hw build_sw

# --------------------------------------------------------------------
# Hardware build (AIE + custom PL kernel + link/package)
# --------------------------------------------------------------------

build_hw: aie custom_pl hw

# AIE graph generates libadf.a
aie:
	$(ECHO) "===> [AIE] Building AIE graph (libadf.a) for TARGET=$(TARGET)"
	$(MAKE) -C aie_overlay clean
	$(MAKE) -C aie_overlay aie_compile TARGET=$(TARGET) PLATFORM=$(PLATFORM)

# Custom PL kernels (movers)
custom_pl:
	$(ECHO) "===> [PL] Building custom PL kernels"
	$(MAKE) -C Custom_Pl_Kernels clean
	$(MAKE) -C Custom_Pl_Kernels all TARGET=$(TARGET) PLATFORM=$(PLATFORM)

# Top-level hardware integration for overlay_$(TARGET).xclbin
hw:
	$(ECHO) "===> [HW] Linking & packaging (v++) for TARGET=$(TARGET)"
	$(MAKE) -C hw clean
	$(MAKE) -C hw all TARGET=$(TARGET) PLATFORM=$(PLATFORM)

# --------------------------------------------------------------------
# Software / host build
# --------------------------------------------------------------------

build_sw: sw

sw:
	$(ECHO) "===> [SW] Building host application"
	$(MAKE) -C sw all TARGET=$(TARGET) PLATFORM=$(PLATFORM)

# --------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------

clean_hw:
	$(ECHO) "===> Cleaning hardware sub-projects"
	-$(MAKE) -C aie_overlay clean
	-$(MAKE) -C Custom_Pl_Kernels clean
	-$(MAKE) -C hw clean

clean_sw:
	$(ECHO) "===> Cleaning software (host)"
	-$(MAKE) -C sw clean

clean_all: clean_hw clean_sw

clean: clean_all

# --------------------------------------------------------------------
# AIE-only + early-link metrics (perf, power, resources)
# --------------------------------------------------------------------
# Usage:
#   make AIELink              # Uses default TARGET=hw
#   make AIELink TARGET=hw_emu
#
# Steps:
#   1) Build AIE graph (libadf.a) via 'aie' target.
#   2) Build PL movers (feat/weight/output) via 'custom_pl' target.
#   3) Run aiesimulator with profiling to get cycle-level perf.
#   4) Run an early v++ --link into ./hw_metrics with --save-temps
#      and --report_level 2. Even if placement fails, synth-level
#      util/power reports are preserved.
#
AIELink: aie custom_pl
	$(ECHO) "===> [AIELINK] Running AIE simulation with profiling"
	cd aie_overlay && aiesimulator --pkg-dir=./Work --profile || true

	$(ECHO) "===> [AIELINK] Early v++ --link for synth/util/power metrics"
	$(ECHO) "      (Placement/route errors will be ignored; reports still kept)"
	mkdir -p hw_metrics
	cd hw_metrics && v++ -l \
		--platform $(PLATFORM) \
		--save-temps \
		--report_level 2 \
		--config ../hw/xclbin_overlay.cfg \
		-o blaze_pose_heads_metrics.xsa \
		../aie_overlay/libadf.a \
		../Custom_Pl_Kernels/feat_mover.xo \
		../Custom_Pl_Kernels/weight_mover.xo \
		../Custom_Pl_Kernels/output_mover.xo || true

	$(ECHO) "===> [AIELINK] Done."
	$(ECHO) "  AIE perf:     aie_overlay/aiesimulator_output/*.aierun_summary"
	$(ECHO) "  Link summary: hw_metrics/link_summary.json (open in vitis_analyzer)"
	$(ECHO) "  Vivado util:  hw_metrics/_x/link/vivado/vpl/prj/prj.runs/*_synth_1/*.rpt"

# --------------------------------------------------------------------
# forMetrics: minimal AIE-only hw_emu overlay (recreates minimal_aie flow)
# --------------------------------------------------------------------
# Usage:
#   make forMetrics
#
# Steps:
#   1) Rebuild AIE graph (libadf.a) for hw_emu.
#   2) Run AIE-only v++ -l -t hw_emu (no PL movers) into ./minimal_aie_metrics.
#   3) Run v++ -p -t hw_emu (with libadf.a) to get a minimal hw_emu xclbin.
#
forMetrics:
	$(ECHO) "===> [METRICS] Building AIE graph (libadf.a) for TARGET=hw_emu"
	$(MAKE) -C aie_overlay clean
	$(MAKE) -C aie_overlay aie_compile TARGET=hw_emu PLATFORM=$(PLATFORM)

	$(ECHO) "===> [METRICS] AIE-only link (hw_emu) for minimal metrics overlay"
	mkdir -p minimal_aie_metrics
	cd minimal_aie_metrics && v++ -l -t hw_emu \
		--platform $(PLATFORM) \
		--save-temps \
		--config ../hw/xclbin_overlay.cfg \
		-o minimal_aie_hw_emu.xsa \
		../aie_overlay/libadf.a

	$(ECHO) "===> [METRICS] AIE-only package (hw_emu) with AIE metadata"
	cd minimal_aie_metrics && v++ -p -t hw_emu \
		--platform $(PLATFORM) \
		--save-temps \
		-o minimal_aie_hw_emu.xclbin \
		../aie_overlay/libadf.a \
		minimal_aie_hw_emu.xsa

	$(ECHO) "===> [METRICS] Done."
	$(ECHO) "  XSA:    minimal_aie_metrics/minimal_aie_hw_emu.xsa"
	$(ECHO) "  XCLBIN: minimal_aie_metrics/minimal_aie_hw_emu.xclbin"
