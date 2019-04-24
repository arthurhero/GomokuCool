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

#define BOARD_DIM 10
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
    bool *host; //is this user the host?
    bool *myturn; //is this user's turn?
    bool *bracket; //draw bracket?
    int *status; //running or host win or guest win or draw
    int *cur_c;
    int *cur_r;
    int *op_c;  // where the opponent placed a piece
    int *op_r;
} game_stat_s;

// struct for read input
typedef struct user_input {
    int *status; //our status
    int *input;
} user_input_s;

// structs for opponent inputs
typedef struct op_info {
    int *status; //our status
    int *r;
    int *c;
    int *op_status;
    bool *offline;
} op_info_s;

// structs for send inputs
typedef struct send_info {
    int r;
    int c;
    int status;
} send_info_s;

#endif
