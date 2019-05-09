#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "info.h"
#include "gpu.h"

// A board of an array of BOARD_DIM *BOARD_DIM cells
typedef struct board {
  int cells[BOARD_DIM * BOARD_DIM];
} board_t;

// Integer to keep track of current state
__device__ int status_d = RUNNING;

// Check each cell as the center of its respective 5-piece-line
__global__ void check_cell(board_t* board) {
  int x = threadIdx.x;
  int y = threadIdx.y;

  // Calculate current cell id in the board
  int cell_id = y * BOARD_DIM + x;

  // Get the current cell
  int cur_cell = board->cells[cell_id];

  // Flag the current cell as checked 
  // 1 - unchecked
  // 0 - checked
  int cur_check = 1;

  // Keep checking the cell until all cells have been checked
  while (__syncthreads_or(cur_check) != 0) {  
    
    // turn the flag as checked
    cur_check = 0;

    // Skip if the cell is blank
    if (cur_cell == 0) {
      continue;
    }

    // Check row success
    int row_s = cur_cell;
    for (int c = x - 2; c <= x + 2; c++) {
      if (c < 0 || c >= BOARD_DIM ||
          board->cells[y * BOARD_DIM + c] != cur_cell) {
        row_s = 0;
      }
    }

    // Check col success
    int col_s = cur_cell;
    for (int r = y - 2; r <= y + 2; r++) {
      if (r < 0 || r >= BOARD_DIM ||
          board->cells[r * BOARD_DIM + x] != cur_cell) {
        col_s = 0;
      }
    }

    // Check / success
    int rl_s = cur_cell;
    for (int i = -2; i <= 2; i++) {
      int c = x + i;
      int r = y - i;
      if (c < 0 || c >= BOARD_DIM ||
          r < 0 || r >= BOARD_DIM ||
          board->cells[r * BOARD_DIM + c] != cur_cell) {
        rl_s = 0;
      }
    }

    // Check \ success
    int lr_s = cur_cell;
    for (int i = - 2; i <= 2; i++) {
      int c = x + i;
      int r = y + i;
      if (c < 0 || c >= BOARD_DIM ||
          r < 0 || r >= BOARD_DIM ||
          board->cells[r * BOARD_DIM + c] != cur_cell) {
        lr_s = 0;
      }
    }

    // Check board complete
    int complete = 4;
    for (int r = 0; r < BOARD_DIM; r++) {
      for(int c = 0; c < BOARD_DIM; c++) {
        if (board->cells[r * BOARD_DIM + c] == 0) {
          complete = 0;
        }
      }
    }

    // Compile results
    int cur_ret = row_s | col_s | rl_s | lr_s | complete;
    
    // Update status_d
    if (complete == 4) {
      status_d = complete;
      return;
    } else if (cur_ret != 0) {
      status_d = cur_cell;
      return;
    }
  }
  return;
}

// Check if the game has a winner
void check_board(int** raw_board, int* res) {
  // Parse raw board into board struct
  board_t* board = (board_t*) malloc(sizeof(board_t));
  for(int i = 0; i < BOARD_DIM; i++) {
    for(int j = 0; j < BOARD_DIM; j++) {
      board->cells[i*BOARD_DIM + j] = raw_board[i][j];
    }
  }

  // Malloc memory in gpu
  board_t* gpu_board;
  if (cudaMalloc(&gpu_board, sizeof(board_t)) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate the board\n");
    exit(2);
  }

  // Copy board to gpus
  if(cudaMemcpy(gpu_board, board, sizeof(board_t), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy board to the GPU\n");
    exit(2);
  }

  // Solve the boards
  check_cell<<<1,dim3(BOARD_DIM, BOARD_DIM)>>>(gpu_board);

  // Wait until it is finished
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    exit(2);
  }

  // Update status
  int status_h;
  cudaMemcpyFromSymbol(&status_h, status_d, sizeof(status_d), 0, cudaMemcpyDeviceToHost);

  // Free the gpu memory
  cudaFree(gpu_board);
  *res = status_h;
}

