#include <stdbool.h>
#include <stdio.h>
ffline=false;
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "socket.h"

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
static void get_opponent_input(int client_socket_fd, int *r, int *c, bool *offline) {
    ssize_t rc;
    rc=read(client_socket_fd, r, sizeof(int));
    if (rc==-1 || rc==0) {
        //if the opponent is offline
        close(client_socket_fd);
        *offline=true;
        return;
    }
    // if the opponent quitted
    if (*r==-1) {
        close(client_socket_fd);
        *offline=true;
        return;
    }
    rc=read(client_socket_fd, c, sizeof(int));
    if (rc==-1 || rc==0) {
        close(client_socket_fd);
        *offline=true;
        return;
    }
    // if the opponent quitted
    if (*c==-1) {
        close(client_socket_fd);
        *offline=true;
        return;
    }
    *offline=false;
    return;
}

//send row and col number to opponent
// if we quit, send -1
// if cannot send, set offline to true
static void send_input(int client_fd, int r, int c, bool *offline) {
    ssize_t rc;
    rc=write(client_fd,&r,sizeof(int));
    if (rc==-1) {
        close(client_fd);
        *offline=true;
        return;
    }
    //if we decided to quit
    if (r==-1) {
        close(client_fd);
        return;
    }
    rc=write(client_fd,&c,sizeof(int));
    if (rc==-1) {
        close(client_fd);
        *offline=true;
        return;
    }
    *offline=false;
    return;
}
