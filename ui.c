#include <curses.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

//#include "info.h"


//int board[BOARD_DIM][BOARD_DIM];


/**
 * Initialize the home scene (welcome screen), programs should call
 * this at the beginning if the user initialize a game room.
 */
/*
void init_board_params(int board_dim){
  *BOARD_DIM = board_dim;
  
  // Initialize board array of int pointers
  board = (int**)malloc(BOARD_DIM * sizeof(int*));

  // Fill in board pointers (to int arrays)
  for(int i = 0; i < BOARD_DIM; i++) {
    board[i] = (int*)malloc(BOARD_DIM * sizeof(int));
  }

  // Fill in the board with BLANK(0)
  for(int i = 0; i < BOARD_DIM; i++) {
    for(int j = 0; j < BOARD_DIM; j++) {
      board[i][j] = BLANK;
    }
  }
} 
*/


/**
 * Convert a board row number to a screen position
 * \param   row   The board row number to convert
 * \return        A corresponding row number for the ncurses screen
 */
int screen_row(int row) {
  return 2 + row;
}

/**
 * Convert a board column number to a screen position
 * \param   col   The board column number to convert
 * \return        A corresponding column number for the ncurses screen
 */
int screen_col(int col) {
  return 2 + col;
}

/**
 * Initialize the home scene (welcome screen), programs should call
 * this at the beginning if the user initialize a game room.
 */
void init_home(int port_num){  
  // Print port num message
  move(2, 0);
  printw("You opened game room at port:");

  // Print port num
  char number[10];
  sprintf(number, "%d", port_num);
  printw(number);

  // Print start board
  int start_row = 4;
  move(start_row, 0);
  printw(" ---------------------------- ");
  move(++start_row, 0);
  printw("|                            |");
  move(++start_row, 0);
  printw("|        GOMOCOOL GAME       |");
  move(++start_row, 0);
  printw("|                            |");
  move(++start_row, 0);
  printw(" ---------------------------- ");

  // Print waiting message
  move(++start_row, 0);
  printw("Waiting for connection. . . ");
  
  // Refresh the display
  refresh();
}

/**
 * Initialize the board, by printing the title and edges.
 */
void init_board() {
  // Print Title Line
  /*
  move(screen_row(-2), screen_col(BOARD_DIM/2 - 5));
  addch(ACS_DIAMOND);
  addch(ACS_DIAMOND);
  printw(" Worm! ");
  addch(ACS_DIAMOND);
  addch(ACS_DIAMOND);
  */
}

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

/**
 * Show a game over message & winner information, and wait for a key press.
 */
void end_game(int winner) {
  mvprintw(screen_row(BOARD_DIM/2)-2, screen_col(BOARD_DIM/2)-6, "            ");
  mvprintw(screen_row(BOARD_DIM/2)-1, screen_col(BOARD_DIM/2)-6, " Game Over! ");

  // Print winner information
  if(winner == HOST_WIN){
    mvprintw(screen_row(BOARD_DIM/2),   screen_col(BOARD_DIM/2)-6, " HOST WIN! ");
  }else if(winner == GUEST_WIN){
    mvprintw(screen_row(BOARD_DIM/2),   screen_col(BOARD_DIM/2)-6, " GUEST WIN!");
  }else if(winner == DRAW){
    mvprintw(screen_row(BOARD_DIM/2),   screen_col(BOARD_DIM/2)-6, "   DRAW!   ");
  }

  mvprintw(screen_row(BOARD_DIM/2)+1, screen_col(BOARD_DIM/2)-6, "            ");
  mvprintw(screen_row(BOARD_DIM/2)+2, screen_col(BOARD_DIM/2)-11, "Press any key to exit.");
  refresh();
  timeout(-1);
  task_readchar();
}

int main(){
  // Initialize the ncurses window
  WINDOW* mainwin = initscr();
  if(mainwin == NULL) {
    fprintf(stderr, "Error initializing ncurses.\n");
    exit(2);
  }
  
  init_home(12345);

  // Clean up window
  //delwin(mainwin);
  //endwin();
  
  return 0;
}
