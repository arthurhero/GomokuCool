CC := clang
CFLAGS := -g -Wall -Wno-deprecated-declarations -Werror

all: gomoku 

clean:
	rm -rf ui ui.dSYM

gomoku: gomoku.c gpu.cu ui.c
	nvcc -g -c gpu.cu
	$(CC) $(CFLAGS) -c ui.c
	$(CC) $(CFLAGS) -o gomoku gpu.o ui.o -L/usr/local/cuda-9.2/lib64 -lcudart -lcuda -lncurses

ui: ui.c
	$(CC) $(CFLAGS) -o ui ui.c -lncurses
