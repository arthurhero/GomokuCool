#include <curses.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

#include "info.h"
#include "network.h"
#include "gpu.h"

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
void init_board(bool host) {
  // Erase greeting prompt First
  erase();

  // Print Title Line
  move(screen_row(-2), screen_col((BOARD_DIM+1)*4/2 - 11));
  addch(ACS_DIAMOND);
  addch(ACS_DIAMOND);
  printw(" GOMOCOOL! ");
  addch(ACS_DIAMOND);
  addch(ACS_DIAMOND);

  // Print Colomn Indicators
  char cur_col = '0';
  move(screen_row(0), screen_col(1));
  addch(cur_col);
  for(int i = 1; i < BOARD_DIM; i++){
    addch(' ');
    addch(' ');
    addch(' ');
    addch(cur_col+i);
  }

  // Print rows
  int cur_row_num = 0;
  char cur_row_char = '0';
  for(int i = 0; i < BOARD_DIM; i++){
    // Move to the start of a row
    move(screen_row(++cur_row_num), screen_col(0));

    // Draw the upper boarder
    for(int i = 0; i < BOARD_DIM; i++ ){
      addch(' ');
      addch('-');
      addch('-');
      addch('-');
    }

    // Move to the cell Line
    move(screen_row(++cur_row_num), screen_col(-2));

    // Add row Indicator
    addch(cur_row_char);
    addch(' ');
    addch('|');

    // Draw the cells
    for(int i = 0; i < BOARD_DIM; i++ ){
      addch(' ');
      addch(' ');
      addch(' ');
      addch('|');
    }

    // Update values
    cur_row_char++;
  }

  // Print the bottom boarder
  //  cur_row_num += 0;
  move(screen_row(++cur_row_num), screen_col(0));
  for(int i = 0; i < BOARD_DIM; i++ ){
    addch(' ');
    addch('-');
    addch('-');
    addch('-');
  }
  refresh();
}

/**
 * Draw bracket
 */
void draw_bracket(int prev_col, int prev_row, int cur_col, int cur_row){
  //First, move to the previous location
  move(screen_row(prev_row*2 >= 2? prev_row*2 +2 : 2),screen_col((prev_col+1)*4-3));

  //Next, draw blanks to cover the current [] that is there
  addch(' ');
  move(screen_row(prev_row*2 >= 2? prev_row*2 +2 : 2),screen_col((prev_col+1)*4-1));
  addch(' ');

  move(screen_row(cur_row*2 >= 2? cur_row*2 +2: 2),screen_col((cur_col+1)*4-3));

  // Draw bracket
  addch('[');
  move(screen_row(cur_row*2 >= 2 ? cur_row*2 +2: 2),screen_col((cur_col+1)*4-1));
  addch(']');

  refresh();
}

/**
 * Draw piece
 */
void draw_piece(int cur_col, int cur_row, char piece){
  // Draw piece
  move(screen_row(cur_row*2 >= 2? cur_row*2 +2 : 2),screen_col((cur_col+1)*4-3));

  addch(' ');
  addch(piece);
  addch(' ');

  refresh();
}

/**
 * Run in a thread to process user input.
 */
void* read_input(void* stat){
  game_stat_s* game_stat = (game_stat_s*)stat;

  // Draw Current bracket
  if (*game_stat->host) {
    draw_bracket(*game_stat->cur_c, *game_stat->cur_r, *game_stat->cur_c, *game_stat->cur_r);
  }

  // Draw current status
  move(screen_row(BOARD_DIM*2 + 3),screen_col(4*BOARD_DIM/2 - 8));
  if(*game_stat->myturn){
    printw("Your Turn!");
  }else{
    printw("Opponent's Turn");
  }


  //Should I change it here?
  while(*game_stat->status == RUNNING) {
    // Update game status
    check_board(game_stat->board, game_stat->status);
    if (*game_stat->status != RUNNING) {
      break;
    }

    if(*game_stat->myturn){
      // Update Turn 
      move(screen_row(BOARD_DIM*2 + 3),screen_col(4*BOARD_DIM/2 - 8));
      printw("               ");
      move(screen_row(BOARD_DIM*2 + 3),screen_col(4*BOARD_DIM/2 - 8));
      if (*game_stat->myturn) {
        printw("Your Turn");
      } else {
        printw("Opponent's Turn");
      }  
      // Read a character, potentially blocking this thread until a key is pressed
      int key = (int) getch();

      // Make sure the input was read correctly
      if(key == ERR) {
        *game_stat->status = WAITING;
        send_input(*game_stat->client_fd, *game_stat->cur_r, *game_stat->cur_c, *game_stat->status);
        fprintf(stderr, "ERROR READING INPUT\n");
        break;
      }

      // Draw Current bracket
      draw_bracket(*game_stat->cur_c, *game_stat->cur_r, *game_stat->cur_c, *game_stat->cur_r);

      // Handle the key press
      if(key == KEY_UP && *game_stat->cur_r != 0) {
        *game_stat->cur_r -= 1;
        draw_bracket(*game_stat->cur_c, *game_stat->cur_r+1, *game_stat->cur_c, *game_stat->cur_r);
      } else if(key == KEY_RIGHT && *game_stat->cur_c != (BOARD_DIM-1)) {
        *game_stat->cur_c += 1;
        draw_bracket(*game_stat->cur_c-1, *game_stat->cur_r, *game_stat->cur_c, *game_stat->cur_r);
      } else if(key == KEY_DOWN && *game_stat->cur_r != (BOARD_DIM-1)) {
        *game_stat->cur_r += 1;
        draw_bracket(*game_stat->cur_c, *game_stat->cur_r-1, *game_stat->cur_c, *game_stat->cur_r);
      } else if(key == KEY_LEFT && *game_stat->cur_c != 0) {
        *game_stat->cur_c -= 1;
        draw_bracket(*game_stat->cur_c+1, *game_stat->cur_r, *game_stat->cur_c, *game_stat->cur_r);
      } else if(key == 'q') {
        *game_stat->status = WAITING;
        send_input(*game_stat->client_fd, *game_stat->cur_r, *game_stat->cur_c, *game_stat->status);
        break;
      } else if(key == '\n'){
        // Skip already occupied cell
        if (game_stat->board[*game_stat->cur_r][*game_stat->cur_c] != BLANK) {
          continue;
        }
        // Draw the piece we just placed
        char piece = *game_stat->host? 'o': '@';
        draw_piece(*game_stat->cur_c, *game_stat->cur_r, piece);

        // Update our board
        int player = *game_stat->host? HOST: GUEST;
        game_stat->board[*game_stat->cur_r][*game_stat->cur_c] = player;

        // Send position and result
        if(send_input(*game_stat->client_fd, *game_stat->cur_r, *game_stat->cur_c, *game_stat->status) == -1){
          *game_stat->status = WAITING;
        }

        *game_stat->myturn = false;

        // Signal input_cv
        pthread_mutex_lock(game_stat->input_m);
        pthread_cond_signal(game_stat->input_cv);
        pthread_mutex_unlock(game_stat->input_m);

        // Update Turn 
        move(screen_row(BOARD_DIM*2 + 3),screen_col(4*BOARD_DIM/2 - 8));
        printw("   Your Turn   ");
        refresh();
      }
    }else{
      // Update Turn 
      move(screen_row(BOARD_DIM*2 + 3),screen_col(4*BOARD_DIM/2 - 8));
      printw(" Opponent Turn ");

      refresh();
      // Wait for opponent to make decision
      pthread_mutex_lock(game_stat->oppo_m);
      while(*game_stat->myturn == false){
        pthread_cond_wait(game_stat->oppo_cv, game_stat->oppo_m);
      }
      pthread_mutex_unlock(game_stat->oppo_m);

      continue;
    }


  }
  // Signal the game over lock
  pthread_mutex_lock(game_stat->over_m);
  pthread_cond_signal(game_stat->over_cv);
  pthread_mutex_unlock(game_stat->over_m);
  return NULL;
}


/**
 * Show a game over message & winner information, and wait for a key press.
 */
void end_game(int winner) {
  erase();
  mvprintw(screen_row(BOARD_DIM/2)-2, screen_col(BOARD_DIM/2)-6, "            ");
  mvprintw(screen_row(BOARD_DIM/2)-1, screen_col(BOARD_DIM/2)-6, " Game Over! ");

  // Print winner information
  if(winner == HOST_WIN){
    mvprintw(screen_row(BOARD_DIM/2),   screen_col(BOARD_DIM/2)-6, " HOST WIN! ");
  }else if(winner == GUEST_WIN){
    mvprintw(screen_row(BOARD_DIM/2),   screen_col(BOARD_DIM/2)-6, " GUEST WIN!");
  }else if(winner == DRAW){
    mvprintw(screen_row(BOARD_DIM/2),   screen_col(BOARD_DIM/2)-6, "   DRAW!   ");
  }else if(winner == WAITING){
    mvprintw(screen_row(BOARD_DIM/2),   screen_col(BOARD_DIM/2)-6, " NET ERROR!");
  }

  mvprintw(screen_row(BOARD_DIM/2)+1, screen_col(BOARD_DIM/2)-6, "            ");
  refresh();
  timeout(-1);
}

