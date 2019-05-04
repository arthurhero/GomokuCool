#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "info.h"
#include "gpu.h"

typedef struct board {
  int cells[BOARD_DIM * BOARD_DIM];
} board_t;

__device__ int status_d = 0;

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
    cur_check = 0;
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
    for (int r = x - 2; r <= x + 2; r++) {
      if (r < 0 || r >= BOARD_DIM ||
          board->cells[r * BOARD_DIM + x] != cur_cell) {
        col_s = 0;
      }
    }

    // Check / success
    int rl_s = cur_cell;
    for (int i = - 2; i <= 2; i++) {
      int c = x + i;
      int r = x - i;
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
      int r = x + i;
      if (c < 0 || c >= BOARD_DIM ||
          r < 0 || r >= BOARD_DIM ||
          board->cells[r * BOARD_DIM + c] != cur_cell) {
        lr_s = 0;
      }
    }

    // Check complete
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
    if (complete == 4) {
      status_d = complete;
      return;
    } else if (cur_ret != 0) {
      status_d = cur_cell;
      printf("cur cell is (%d, %d, %d, %d)\n", row_s, col_s, rl_s, lr_s);
      return;
    }
  }
  return;
}


void check_board(int** board, int* res) {
  // Malloc memory in gpu
  board_t* gpu_board;
  if (cudaMalloc(&gpu_board, sizeof(board_t)) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate the board\n");
    exit(2);
  }

  // Copy board to gpus
  if(cudaMemcpy(gpu_board, *board, sizeof(board_t), cudaMemcpyHostToDevice) != cudaSuccess) {
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
  printf("The results shows that player %d wins\n", status_h);

  // Free the gpu memory
  cudaFree(gpu_board);
  *res = status_h;
}


//Test
int main(int argc, char** argv) {
  int test[BOARD_DIM][BOARD_DIM] = {0};
  test[1][1] = 2;
  test[2][2] = 2;
  test[3][3] = 2;
  test[4][4] = 2;
  //test[5][5] = 2;
  int* p = (int*) test;
  int res;
  check_board(&p, &res);
  printf("this is a test in main function. Winner is %d\n", res);
  return 0;
}

