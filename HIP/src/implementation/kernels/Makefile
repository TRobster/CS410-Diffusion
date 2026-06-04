HIPCC    := hipcc
ARCH     ?=
HIPFLAGS := -std=c++17 -O2 $(if $(ARCH),--offload-arch=$(ARCH),)

INC     := ../../../include
OBJ_DIR := obj
LIB     := libheat.a

SRCS := kernel_utils.cc kernels.cc
OBJS := $(SRCS:%.cc=$(OBJ_DIR)/%.o)

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(OBJ_DIR)/%.o: %.cc | $(OBJ_DIR)
	$(HIPCC) $(HIPFLAGS) -I$(INC) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

help:
	@echo "For more info on compiling for specific architecture look in HIP/info/architecture_flags.txt"

clean:
	rm -rf $(OBJ_DIR) $(LIB)
