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

#include "util.h"
#include "gpu.h"
#include "info.h"
#include "network.h"


/**
 * In-memory representation of the game board
 * 0 represents a white piece
 * 1 represents a black piece
 */
int board[BOARD_DIM][BOARD_DIM];

// status of running or winning or losing
int status = WAITING;

// Are you the host? (host always use white piece and start first)
bool host;
// Is it my turn?
bool myturn;

// current location of user cursor 
int cur_r=BOARD_DIM/2,cur_c=BOARD_DIM/2;

// Entry point: Set up the game, create jobs, then run the scheduler
int main(void) {
  int rc;

  // Initialize the ncurses window
  WINDOW* mainwin = initscr();
  if(mainwin == NULL) {
    fprintf(stderr, "Error initializing ncurses.\n");
    exit(2);
  }
  
  // Seed random number generator with the time in milliseconds
  srand(time_ms());
  
  noecho();               // Don't print keys when pressed
  keypad(mainwin, true);  // Support arrow keys
  nodelay(mainwin, true); // Non-blocking keyboard access

  // Check the argument number is correct
  if(argc != 1 && argc != 3) {
      fprintf(stderr, "Usage: %s [<host name> <port number>]\n", argv[0]);
      exit(1);
  }

  int server_socket_fd;
  int server_port;
  int socket_fd;

  // Check whether we want to connect to other game room
  if(argc == 3) {
      host=false;
      // Unpack arguments
      char* peer_hostname = argv[1];
      unsigned short peer_port = atoi(argv[2]);
      join_game_room(peer_hostname, peer_port, &socket_fd);
  } else {
      host=true;
      open_game_room(&server_port, &server_socket_fd);
      // Initialize the home game display
      init_home(server_port);
      // Accept connection
      accept_connection(server_socket_fd, &socket_fd);
  }

  // Zero out the board contents
  memset(board, BLANK, BOARD_DIM*BOARD_DIM*sizeof(int));

  if (host) {
      myturn=true;
  } else {
      myturn=false;
  }

  // locks and cond variables
  pthread_cond_t over_cv = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t over_m = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t input_cv = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t input_m = PTHREAD_MUTEX_INITIALIZER;

  // Display the game board
  init_board();

  // Thread handles for each of the game threads
  pthread_t update_board_thread;
  pthread_t read_input_thread;
  pthread_t read_opponent_thread; 

  // variables for storing input opponent info
  int myinput;
  int op_r;
  int op_c;
  int op_stat;
  bool op_offline;

  // input for update board
  game_stat_s *gstat = (game_stat_s *)malloc(sizeof(game_stat_s));
  gstat->host = &host;
  gstat->myturn = &myturn;
  gstat->status = &status;
  gstat->board = board;
  gstat->cur_c = &cur_c;
  gstat->cur_r = &cur_r;
  gstat->op_c = &op_c;
  gstat->op_r = &op_r;
  gstat->over_cv = &over_cv;
  gstat->over_m = &over_m;
  gstat->input_cv = &input_cv;
  gstat->input_m = &input_m;
  gstat->oppo_cv = &oppo_cv;
  gstat->oppo_m = &oppo_m;
  gstat->client_fd = &socket_fd;

  // create thread for update board
  rc = pthread_create(&(update_board_thread),NULL,draw_board,gstat);
  if (rc) {
      perror("failed to create thread for update board");
      exit(2);
  }
  // input struct for read input
  user_input_s *user_input = (user_input_s *)malloc(sizeof(user_input_s));
  user_input->status = &status;
  user_input->input = &myinput;
  // create thread for read user input 
  rc = pthread_create(&(read_input_thread),NULL,read_input,user_input);
  if (rc) {
      perror("failed to create thread for read input");
      exit(2);
  }
  // create thread for get opponent info 
  rc = pthread_create(&(read_opponent_thread),NULL,get_opponent_input,opinfo);
  if (rc) {
      perror("failed to create thread for read input");
      exit(2);
  }

  while (status==RUNNING) {
      if (myinput == QUIT) {
          status=WAITING;
          break;
      }
      if (myturn) {
          if (myinput == RIGHT && cur_c<BOARD_DIM-1) cur_c++;
          else if (myinput == LEFT && cur_c>0) cur_c--;
          else if (myinput == UP && cur_r>0) cur_r--;
          else if (myinput == DOWN && cur_r<BOARD_DIM-1) cur_r++;
          else if (myinput == ENTER) {
              if (host) board[cur_r][cur_c]=HOST;
              else board[cur_r][cur_c]=GUEST;
              int result = check_board(board);
              if (result == HOST_WIN || result == GUEST_WIN || result == DRAW) {
                  status = result;
                  break;
              }
          }
          myinput=NONE;
          myturn = false;
      } else {
          if (op_offline) {
              status=WAITING;
              break;
          } else {
              int result = op_status;
              if (result == HOST_WIN || result == GUEST_WIN || result == DRAW) {
                  status = result;
                  break;
              } else {
                  if (host) board[op_r][op_c]=GUEST;
                  else board[op_r][op_c]=HOST;
              }
          }
      }
  }

  end_game(status);
  sleep(5);

  //TODO: if guest, change to host, open game room


  // Clean up window
  delwin(mainwin);
  endwin();

  return 0;
}
