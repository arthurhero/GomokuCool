#ifndef INFO_H
#define INFO_H

// Game status
#define RUNNING 0
#define HOST_WIN 1
#define GUEST_WIN 2
#define DRAW 4
#define WAITING 5 //waiting for an opponent to connect

// Cell status
#define BLANK 0
#define HOST 1
#define GUEST 2

// Board Attribute
#define BOARD_DIM 20

// structs for task methods
typedef struct game_stat {

  // A boolean to mark user identity
  // 1 represents current user is host
  // 2 represents current user is guest
  bool *host; 

  int **board;

  // A boolean to mark if it's user's turn
  bool *myturn; 

  // Status of the game
  int *status; 

  // Coordinates of current user's piece
  int *cur_c;
  int *cur_r;

  // Coordinates of opponent's piece
  int *op_c;  
  int *op_r;

  // Conditional & locks
  pthread_cond_t *over_cv;
  pthread_mutex_t *over_m;
  pthread_cond_t *input_cv;
  pthread_mutex_t *input_m;
  pthread_cond_t *oppo_cv;
  pthread_mutex_t *oppo_m;
  
  // Client socket
  int *client_fd;
} game_stat_s;

#endif
