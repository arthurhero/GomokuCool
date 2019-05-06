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
 * Run in a thread to draw the current state of the game board.

void* draw_board(void* stat){
  game_stat_s* game_stat = (game_stat_s*)stat;

  int toggle_turn = false;
  bool prev_my_turn = *game_stat->myturn;
  int prev_row = *game_stat->cur_c;
  int prev_col = *game_stat->cur_r;

  // Keep updataing the board while the game is running
  while(game_stat->status == RUNNING){

    // Determine if we just changed player
    if(*game_stat->myturn != prev_my_turn) {
      toggle_turn = true;
    }else{
      toggle_turn = false;
    }

    // Update game status
    bool draw_bracket = *game_stat->bracket;
    bool my_turn = *game_stat->myturn;
    int cur_col = *game_stat->cur_c;
    int cur_row = *game_stat->cur_r;

    if(my_turn){
      // If the turn has not been toggled erase the previous [] we drew
      if(!toggle_turn){
        //First, move to the previous location
        move((prev_col+1)*4-1, (prev_row-2)/2);

        //Next, draw blanks to cover the current [] that is there
        addch(' ');
        addch(' ');
        addch(' ');
      }

      // Draw bracket or a piece
      if(draw_bracket){
        // move to the current position in board
        move((cur_col+1)*4-1, (cur_row-2)/2);

        // Draw bracket
        addch('[');
        addch(' ');
        addch(']');

      }else{ // draw piece
        // move to the current position in board
        move((cur_col+1)*4-1, (cur_row-2)/2);

        bool host_piece = *game_stat->host;

        if(host_piece){
          addch('@');
        }else{
          addch('o');
        }

      }
    }

    // Repetitively draw opponent's piece
      int op_col = *game_stat->op_c;
      int op_row = *game_stat->op_r;
      // move to the current position in board
      move((op_c+1)*4, (op_r-2)/2);

      bool host_piece = *game_stat->host;

      // make sure that we are not in the first round (i.e. there is no opponent piece yet)
      if(op_col != -1 && op_row != -1){
      // print piece
      if(host_piece){
        addch('@');
      }else{
        addch('o');
      }
    }

  prev_my_turn = my_turn;
  prev_col = cur_col;
  prev_row = cur_row;
  refresh();
 }

  return NULL;
}
*/

/**
 * Draw bracket
 */
void draw_bracket(int prev_col, int prev_row, int cur_col, int cur_row){
  //First, move to the previous location
  move(screen_row(prev_row*2 > 2? prev_row*2 +2 : 2),screen_col((prev_col+1)*4-3));

  //Next, draw blanks to cover the current [] that is there
  addch(' ');
  addch(' ');
  addch(' ');

  move(screen_row(cur_row*2 > 2? cur_row*2 +2: 2),screen_col((cur_col+1)*4-3));

  // Draw bracket
  addch('[');
  addch(' ');
  addch(']');

  refresh();
}

/**
 * Draw piece
 */
void draw_piece(int cur_col, int cur_row, char piece){
   // Draw piece
   move(screen_row(cur_row*2 > 2? cur_row*2 +2 : 2),screen_col((cur_col+1)*4-3));

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

  //Should I change it here?
  while(*game_stat->status == RUNNING) {
    if(*game_stat->myturn){
    // Read a character, potentially blocking this thread until a key is pressed
    int key = (int) getch();

    // Make sure the input was read correctly
    if(key == ERR) {
      *game_stat->status = WAITING;
      send_input(*game_stat->client_fd, *game_stat->cur_r, *game_stat->cur_c, *game_stat->status);
      fprintf(stderr, "ERROR READING INPUT\n");
      break;
    }

    // Print current player name
    move(screen_row(BOARD_DIM*2 + 3),screen_col(4*BOARD_DIM/2 - 8));
    if(*game_stat->host){
      printw("CURRENT : HOST");
    }else{
      printw("CURRENT : GUEST");
    }

    // Handle the key press
    if(key == UP && *game_stat->cur_r != 0) {
      *game_stat->cur_r -= 1;
      draw_bracket(*game_stat->cur_c, *game_stat->cur_r+1, *game_stat->cur_c, *game_stat->cur_r);
    } else if(key == RIGHT && *game_stat->cur_c != (BOARD_DIM-1)) {
      *game_stat->cur_c += 1;
      draw_bracket(*game_stat->cur_c-1, *game_stat->cur_r, *game_stat->cur_c, *game_stat->cur_r);
    } else if(key == DOWN && *game_stat->cur_r != (BOARD_DIM-1)) {
      *game_stat->cur_r += 1;
      draw_bracket(*game_stat->cur_c, *game_stat->cur_r-1, *game_stat->cur_c, *game_stat->cur_r);
    } else if(key == LEFT && *game_stat->cur_c != 0) {
      *game_stat->cur_c -= 1;
      draw_bracket(*game_stat->cur_c+1, *game_stat->cur_r, *game_stat->cur_c, *game_stat->cur_r);
    } else if(key == 'q') {
      *game_stat->status = WAITING;
      send_input(*game_stat->client_fd, *game_stat->cur_r, *game_stat->cur_c, *game_stat->status);
      break;
    } else if(key == '\n'){
      // Update game status
      //check_board(game_stat->board, game_stat->status);

      char piece = *game_stat->host? 'o': '@';
      draw_piece(*game_stat->cur_c, *game_stat->cur_r, piece);

      // Send position and result
      if(send_input(*game_stat->client_fd, *game_stat->cur_r, *game_stat->cur_c, *game_stat->status) == -1){
        *game_stat->status = WAITING;
      }

      // Signal input_cv
      pthread_mutex_lock(game_stat->input_m);
      pthread_cond_signal(game_stat->input_cv);
      pthread_mutex_unlock(game_stat->input_m);

      *game_stat->myturn = false;

      // Wait for opponent to make decision
      pthread_mutex_lock(game_stat->oppo_m);
      while(*game_stat->myturn == false){
        pthread_cond_wait(game_stat->oppo_cv, game_stat->oppo_m);
      }
      pthread_mutex_unlock(game_stat->oppo_m);

    }
  }else{
  // Print current player name
  move(screen_row(BOARD_DIM*2 + 3),screen_col(4*BOARD_DIM/2 - 8));
  if(*game_stat->host){
    printw("CURRENT : GUEST");
  }else{
    printw("CURRENT : HOST");
  }
    continue;
  }

}
 pthread_mutex_lock(game_stat->over_m);
 pthread_cond_signal(game_stat->over_cv);
 pthread_mutex_lock(game_stat->over_m);
 return NULL;
}


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
  }else if(winner == WAITING){
    mvprintw(screen_row(BOARD_DIM/2),   screen_col(BOARD_DIM/2)-6, " NET ERROR!");
  }

  mvprintw(screen_row(BOARD_DIM/2)+1, screen_col(BOARD_DIM/2)-6, "            ");
  mvprintw(screen_row(BOARD_DIM/2)+2, screen_col(BOARD_DIM/2)-11, "Press any key to exit.");
  refresh();
  timeout(-1);
  //readchar();
}

/*
int main(){
  // Initialize the ncurses window
  WINDOW* mainwin = initscr();
  if(mainwin == NULL) {
    fprintf(stderr, "Error initializing ncurses.\n");
    exit(2);
  }

  //init_home(12345);


  game_stat_s st;
  int col = 5;
  int row = 6;
  bool host = false;

  st.cur_c = &col;
  st.cur_r = &row;
  st.host = &host;

  init_home(2345);
  sleep(2);
  init_board();
  draw_bracket(1,1,2,2);
  sleep(2);
  draw_bracket(2,2,3,4);
  sleep(2);
  //draw_bracket(3,4,11,14);
  //sleep(2);
  //draw_bracket(11,14,7,9);
  //sleep(2);
  draw_piece(2,3,'@');
  sleep(2);
  draw_piece(5,6,'@');
  sleep(2);
  draw_piece(0,0,'o');
  sleep(2);
  //draw_piece(1,14,'o');
  // Clean up window
  delwin(mainwin);
  endwin();

  return 0;
}
*/
