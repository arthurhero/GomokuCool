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

#include "socket.h"
#include "gpu.h"
#include "info.h"
#include "network.h"
#include "util.h"

/**
 * In-memory representation of the game board
 * 0 represents a blank space
 * 1 represents a host piece
 * 2 represents a guest piece
 */
int **board;

// State of the game
int status = WAITING;

// Are you the host? (host always use white piece and start first)
bool host;
// Is it my turn?
bool myturn;

// Current location of user cursor 
int cur_r=BOARD_DIM/2,cur_c=BOARD_DIM/2;

// Location of opponent cursor
int op_c=BOARD_DIM/2, op_r=BOARD_DIM/2;

// Entry point: Set up the game, create jobs, then run the scheduler
int main(int argc, char** argv) {
  int rc;

  //Allocate board
  board = (int **)malloc(BOARD_DIM * sizeof(int*)); 
  for (int i=0; i<BOARD_DIM; i++) 
      board[i] = (int*)calloc(BOARD_DIM, sizeof(int)); 

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

  // Check the argument number is correct
  if(argc != 1 && argc != 3) {
      fprintf(stderr, "Usage: %s [<host name> <port number>]\n", argv[0]);
      exit(1);
  }

  // Declare sockets and port
  int server_socket_fd;
  unsigned short server_port;
  int socket_fd;

  // Check whether we want to connect to other game room
  if(argc == 3) {
      host=false;

      // Unpack arguments
      char* peer_hostname = argv[1];
      unsigned short peer_port = atoi(argv[2]);

      // Join the host's game
      join_game_room(peer_hostname, peer_port, &socket_fd);
  } else {
      host=true;

      // Initialize the home game
      open_game_room(&server_port, &server_socket_fd);

      // Initialize the home game display
      init_home(server_port);
      // Accept connection
      accept_connection(server_socket_fd, &socket_fd);
  }

  // Host play first
  if (host) {
      myturn=true;
  } else {
      myturn=false;
  }

  // locks and cond variables (detailed comment see info.h)
  pthread_cond_t over_cv = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t over_m = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t input_cv = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t input_m = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t oppo_cv = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t oppo_m = PTHREAD_MUTEX_INITIALIZER;

  // Display the game board
  init_board(host);

  //start the game
  status = RUNNING;

  // Thread handles for each of the game threads
  pthread_t read_input_thread;
  pthread_t read_opponent_thread; 

  // Initialize gstat struct for the threads 
  game_stat_s *gstat = (game_stat_s *)malloc(sizeof(game_stat_s));
  gstat->host = &host;
  gstat->board = (int **)board;
  gstat->myturn = &myturn;
  gstat->status = &status;
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

  // Create thread for read user input 
  rc = pthread_create(&(read_input_thread),NULL,read_input,gstat);
  if (rc) {
      perror("failed to create thread for read input");
      exit(2);
  }
  // Create thread for get opponent info 
  rc = pthread_create(&(read_opponent_thread),NULL,get_opponent_input,gstat);
  if (rc) {
      perror("failed to create thread for read opponent info");
      exit(2);
  }

  // Wait for the game to end
  pthread_mutex_lock(&over_m);
  while (status==RUNNING) {
      pthread_cond_wait(&over_cv,&over_m);
  }
  pthread_mutex_unlock(&over_m);

  // Display end game user interface
  end_game(status);
  sleep(5);

  //TODO: if guest, change to host, open game room

  // Clean up window
  delwin(mainwin);
  endwin();

  return 0;
}
