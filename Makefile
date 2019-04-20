CC := clang
CFLAGS := -g -Wall -Wno-deprecated-declarations -Werror

all: ui

clean:
	rm -rf ui ui.dSYM

ui: ui.c
	$(CC) $(CFLAGS) -o ui ui.c -lncurses
