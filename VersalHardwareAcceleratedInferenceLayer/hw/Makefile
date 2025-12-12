# hw/Makefile - Link & package BlazePose pose-head overlay for VCK5000

ECHO      := @echo
VPP       := v++
PLATFORM  ?= /opt/xilinx/platforms/xilinx_vck5000_gen4x8_qdma_2_202220_1/xilinx_vck5000_gen4x8_qdma_2_202220_1.xpfm
TARGET    ?= hw   # can be: hw, hw_emu, sw_emu

# --------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------

# AIE archive from aie_overlay
AIE_ARCHIVE := ../aie_overlay/libadf.a

# Custom PL kernels (data movers)
PL_KRNL_XO := \
  ../Custom_Pl_Kernels/feat_mover.xo \
  ../Custom_Pl_Kernels/weight_mover.xo \
  ../Custom_Pl_Kernels/output_mover.xo

# Connectivity config
XCLBIN_CFG := xclbin_overlay.cfg

# Default outputs (for hw / hw_emu) – keep existing behavior
XSA      := overlay_$(TARGET).xsa
XCLBIN   := overlay_$(TARGET).xclbin

# For sw_emu, put outputs in a separate directory
ifeq ($(TARGET),sw_emu)
OUT_DIR := out_sw_emu
XSA     := $(OUT_DIR)/overlay_sw_emu.xsa
XCLBIN  := $(OUT_DIR)/overlay_sw_emu.xclbin
endif

# --------------------------------------------------------------------
# v++ flags
# --------------------------------------------------------------------

VPP_COMMON_FLAGS := --platform $(PLATFORM) --save-temps

ifeq ($(TARGET),hw_emu)
VPP_LINK_FLAGS    := -l -t hw_emu  $(VPP_COMMON_FLAGS)
VPP_PACKAGE_FLAGS := -p -t hw_emu  $(VPP_COMMON_FLAGS)
else ifeq ($(TARGET),sw_emu)
VPP_LINK_FLAGS    := -l -t sw_emu  $(VPP_COMMON_FLAGS)
VPP_PACKAGE_FLAGS := -p -t sw_emu  $(VPP_COMMON_FLAGS)
else
VPP_LINK_FLAGS    := -l -t hw      $(VPP_COMMON_FLAGS)
VPP_PACKAGE_FLAGS := -p -t hw      $(VPP_COMMON_FLAGS)
endif

# --------------------------------------------------------------------
# Top-level targets
# --------------------------------------------------------------------

.PHONY: all link package clean

all: $(XCLBIN)

# --------------------------------------------------------------------
# Link step: AIE + PL -> .xsa
# --------------------------------------------------------------------

link: $(XSA)

# For sw_emu, ensure OUT_DIR exists before writing XSA
ifeq ($(TARGET),sw_emu)
$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

$(XSA): $(AIE_ARCHIVE) $(PL_KRNL_XO) $(XCLBIN_CFG) | $(OUT_DIR)
	$(ECHO) "===> [HW] Linking overlay for TARGET=$(TARGET)"
	$(VPP) $(VPP_LINK_FLAGS) \
		--config $(XCLBIN_CFG) \
		-o $@ \
		$(AIE_ARCHIVE) \
		$(PL_KRNL_XO)

else

# Original rule for hw / hw_emu (unchanged)
$(XSA): $(AIE_ARCHIVE) $(PL_KRNL_XO) $(XCLBIN_CFG)
	$(ECHO) "===> [HW] Linking overlay for TARGET=$(TARGET)"
	$(VPP) $(VPP_LINK_FLAGS) \
		--config $(XCLBIN_CFG) \
		-o $@ \
		$(AIE_ARCHIVE) \
		$(PL_KRNL_XO)

endif

# --------------------------------------------------------------------
# Package step: .xsa -> .xclbin (must include libadf.a!)
# --------------------------------------------------------------------

package: $(XCLBIN)

$(XCLBIN): $(XSA) $(AIE_ARCHIVE)
	$(ECHO) "===> [HW] Packaging $(TARGET) xclbin (with AIE)"
	$(VPP) $(VPP_PACKAGE_FLAGS) \
		-o $@ \
		$(AIE_ARCHIVE) \
		$<

# --------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------

clean:
	$(ECHO) "===> [HW] Cleaning hw build artifacts"
	$(RM) -r _x .Xil *.log *.jou *.info
	$(RM) -f overlay_hw.xsa overlay_hw_emu.xsa overlay_hw.xclbin overlay_hw_emu.xclbin
	$(RM) -r out_sw_emu
