CC := nvcc 
CFLAGS := -g
#CFLAGS := -g -Wno-deprecated-declarations -Werror

all: gomoku 

clean:
	rm -rf ui ui.dSYM

gomoku: gomoku.cu gpu.cu ui.cu util.cu
	$(CC) $(CFLAGS) $^ -o gomoku -lncurses -lpthread

ui: ui.c
	$(CC) $(CFLAGS) -o ui ui.c -lncurses
