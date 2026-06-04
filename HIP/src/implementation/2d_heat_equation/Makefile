HIPCC    := hipcc
ARCH     ?=
HIPFLAGS := -std=c++17 -O2 $(if $(ARCH),--offload-arch=$(ARCH),)

INC     := ../../../include
OBJ_DIR := obj
LIB     := libheat_equation.a

SRCS := heat_equation.cc
OBJS := $(SRCS:%.cc=$(OBJ_DIR)/%.o)

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(OBJ_DIR)/%.o: %.cc | $(OBJ_DIR)
	$(HIPCC) $(HIPFLAGS) -I$(INC) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(LIB)
