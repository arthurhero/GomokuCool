#ifndef INFO_H
#define INFO_H

// game status
#define RUNNING 0
#define HOST_WIN 1
#define GUEST_WIN 2
#define DRAW 4
#define WAITING 5 //waiting for an opponent to connect

#define BLANK 0
#define HOST 1
#define GUEST 2

#define BOARD_DIM 20
#define BOARD_HEIGHT 2
#define BOARD_WIDTH 4

// Defines user inputs
#define NONE -1
#define UP 0
#define RIGHT 1
#define DOWN 2
#define LEFT 3
#define ENTER 4
#define QUIT 5
#define YES 6
#define NO 7

// structs for task methods
typedef struct game_stat {
    bool *host; //is this user the host? used to deterine the piece pattern, 1 for ishost, 0 for notHost
    int **board;
    bool *myturn; //is this user's turn? 1 stands for yes, 0 for no
    int *status; //running or host win or guest win or draw
    int *cur_c;
    int *cur_r;
    int *op_c;  // where the opponent placed a piece, the value is -1 if we are in the first round
    int *op_r;
    pthread_cond_t *over_cv;
    pthread_mutex_t *over_m;
    pthread_cond_t *input_cv;
    pthread_mutex_t *input_m;
    pthread_cond_t *oppo_cv;
    pthread_mutex_t *oppo_m;
    int *client_fd;
} game_stat_s;

#endif
