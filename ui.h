#ifndef UI_H
#define UI_H

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
void init_home(int port_num);

/**
 * Initialize the board, by printing the title and edges.
 */
void init_board(bool host);

/**
 * Run in a thread to draw the current state of the game board.
 */
void draw_bracket(int prev_col, int prev_row, int cur_col, int cur_row);

void draw_piece(int cur_col, int cur_row, char piece);


/**
 * Run in a thread to process user input.
 */
void* read_input(void* stat);

/**
 * Run in a thread to move the worm around on the board
 */
void end_game(int winner);


#endif
