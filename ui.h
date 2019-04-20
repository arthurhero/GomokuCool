#include <curses.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/**
 * Initialize the home scene (welcome screen), programs should call
 * this at the beginning if the user initialize a game room.
 */
void init_home();

/**
 * Initialize the board, by printing the title and edges.
 */
void init_board();

/**
 * Run in a thread to draw the current state of the game board.
 */
void draw_board();

/**
 * Run in a thread to process user input.
 */
void read_input();

/**
 * Run in a thread to move the worm around on the board
 */
void update_piece();



