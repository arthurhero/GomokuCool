#ifndef NETWORK_H
#define NETWORK_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#include "socket.h"
#include "ui.h"

// Open a server socket and return the port number and server socket fd
static void open_game_room(unsigned short *port, int *server_socket_fd){
    // Open a server socket
    *server_socket_fd = server_socket_open(port);
    if (*server_socket_fd == -1) {
        perror("Server socket was not opened");
        exit(2);
    }

    // Start listening for connections
    if (listen(*server_socket_fd, 1)) {
        perror("listen failed");
        exit(2);
    }
}

// connect to an existing game room
// return the opponent socket fd
static void join_game_room(char *peer_hostname, unsigned short peer_port, int *socket_fd) {
    *socket_fd = socket_connect(peer_hostname, peer_port);
    if (*socket_fd == -1) {
        perror("Failed to connect");
        exit(2);
    }

}

// accept connection from new opponent
// return the opponent port
static void accept_connection(int server_socket_fd, int *client_socket_fd) {
    *client_socket_fd = server_socket_accept(server_socket_fd);
    if (*client_socket_fd == -1) {
        perror("Accept failed");
        exit(2);
    }
    return;
}

// receive opponent input: row and col
// if fail to get input, set offline to true
static void* get_opponent_input(void *arg) {
    game_stat_s *game_stat = (game_stat_s *)arg;
    ssize_t rc;
    while (*game_stat->status==RUNNING) {
        if (!(*game_stat->myturn)) {
            rc=read(*game_stat->client_fd, game_stat->op_r, sizeof(int));
            if (rc==-1 || rc==0) {
                //if the opponent is offline
                close(*game_stat->client_fd);
                *game_stat->status=WAITING;
                break;
            }
            rc=read(*game_stat->client_fd, game_stat->op_c, sizeof(int));
            if (rc==-1 || rc==0) {
                close(*game_stat->client_fd);
                *game_stat->status=WAITING;
                break;
            }
            rc=read(*game_stat->client_fd, game_stat->status, sizeof(int));
            if (rc==-1 || rc==0) {
                close(*game_stat->client_fd);
                *game_stat->status=WAITING;
                break;
            }
            //finished reading
            // update and draw the board
            if (*game_stat->status==RUNNING) {
                game_stat->board[*game_stat->op_r][*game_stat->op_c]=GUEST;
                draw_piece(*game_stat->op_c,*game_stat->op_r,'@');
            }
            //signal the user input thread
            *game_stat->myturn = true;
            pthread_mutex_lock(game_stat->oppo_m);
            pthread_cond_signal(game_stat->oppo_cv);
            pthread_mutex_unlock(game_stat->oppo_m);
            //wait for opponent's turn
            pthread_mutex_lock(game_stat->input_m);
            while (*game_stat->myturn) {
                pthread_cond_wait(game_stat->input_cv,game_stat->input_m);
            }
            pthread_mutex_unlock(game_stat->input_m);
        } else {
            continue;
        }
    }
    return NULL;
}

//send row and col number to opponent
// if we quit, send -1
// if cannot send, we think opponent quitted, return -1
// On success, return 0 
static int send_input(int client_fd, int r, int c, int status) {
    ssize_t rc;
    rc=write(client_fd,&r,sizeof(int));
    if (rc==-1) {
        close(client_fd);
        return -1;
    }
    rc=write(client_fd,&c,sizeof(int));
    if (rc==-1) {
        close(client_fd);
        return -1;
    }
    rc=write(client_fd,&status,sizeof(int));
    if (rc == -1) {
        close(client_fd);
        return -1;
    }
    return 0;
}

#endif
