# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ECHO=@echo

.PHONY: help all clean

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""

# Software build for XRT Native API code

# C++ flags
CXXFLAGS := -std=c++17 -Wno-deprecated-declarations
CXXFLAGS += -I$(XILINX_XRT)/include

# Linker flags
LDFLAGS := -L$(XILINX_XRT)/lib -lxrt_coreutil

# Host executable name
EXECUTABLE := host_overlay.exe

# Host source(s) in this sw/ directory
HOST_SRCS := ./host_overlay.cpp

all: build_sw

build_sw: $(EXECUTABLE)

$(EXECUTABLE): $(HOST_SRCS)
	$(CXX) -o $(EXECUTABLE) $^ $(CXXFLAGS) $(LDFLAGS)
	@rm -f ./overlay_hw.xclbin
	@rm -f ./overlay_hw_emu.xclbin
	@ln -s ../hw/overlay_hw.xclbin
	@ln -s ../hw/overlay_hw_emu.xclbin

#clean up

clean:
	$(RM) -r _x .Xil *.ltx *.log *.jou *.info host_overlay.exe *.xo *.xo.* *.str *.xclbin
