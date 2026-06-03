CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

OBJ_DIR := obj
LIB     := libpgm.a

SRCS := pgm_filereader.cc
OBJS := $(SRCS:%.cc=$(OBJ_DIR)/%.o)

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(OBJ_DIR)/%.o: %.cc pgm_filereader.h | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(LIB)
