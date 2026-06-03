HIPCC    := hipcc
HIPFLAGS := -std=c++17 -O2

ROOT     := ../../..
INC      := ../../include
IMPL_DIR := ../implementation
DATA_DIR := $(ROOT)/data_pipeline

OBJ_DIR := obj
BIN     := heat_test

SRCS := main.cc correctness.cc timing.cc
OBJS := $(SRCS:%.cc=$(OBJ_DIR)/%.o)

IMPL_LIB := $(IMPL_DIR)/libheat.a
DATA_LIB := $(DATA_DIR)/libpgm.a

.PHONY: all clean $(IMPL_LIB) $(DATA_LIB)

all: $(IMPL_LIB) $(DATA_LIB) $(BIN)

$(IMPL_LIB):
	$(MAKE) -C $(IMPL_DIR)

$(DATA_LIB):
	$(MAKE) -C $(DATA_DIR)

$(BIN): $(OBJS) $(IMPL_LIB) $(DATA_LIB)
	$(HIPCC) $(HIPFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: %.cc | $(OBJ_DIR)
	$(HIPCC) $(HIPFLAGS) -I$(INC) -I$(DATA_DIR) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN)
