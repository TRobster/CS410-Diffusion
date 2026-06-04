HIPCC    := hipcc
HIPFLAGS := -std=c++17 -O2

ROOT        := ../../../..
INC         := ../../../include
KERNELS_DIR := ../../implementation/kernels
SOLVER_DIR  := ../../implementation/2d_heat_equation
DATA_DIR    := $(ROOT)/data_pipeline

OBJ_DIR := obj
BIN     := solver_test

SRCS := main.cc correctness.cc timing.cc
OBJS := $(SRCS:%.cc=$(OBJ_DIR)/%.o)

KERNELS_LIB := $(KERNELS_DIR)/libheat.a
SOLVER_LIB  := $(SOLVER_DIR)/libheat_equation.a
DATA_LIB    := $(DATA_DIR)/libpgm.a

.PHONY: all clean $(KERNELS_LIB) $(SOLVER_LIB) $(DATA_LIB)

all: $(KERNELS_LIB) $(SOLVER_LIB) $(DATA_LIB) $(BIN)

$(KERNELS_LIB):
	$(MAKE) -C $(KERNELS_DIR)

$(SOLVER_LIB):
	$(MAKE) -C $(SOLVER_DIR)

$(DATA_LIB):
	$(MAKE) -C $(DATA_DIR)

$(BIN): $(OBJS) $(KERNELS_LIB) $(SOLVER_LIB) $(DATA_LIB)
	$(HIPCC) $(HIPFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: %.cc | $(OBJ_DIR)
	$(HIPCC) $(HIPFLAGS) -I$(INC) -I$(DATA_DIR) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN)
