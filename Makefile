CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -fopenmp
TARGET = smooth
SRC = main.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o a.out output.pgm
