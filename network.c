#include "network.h"

int main(int argc, char** info) {
  if (argc == 1) {
    unsigned short port = 0;
    int server_socket_fd;
    int client_socket_fd;
    open_game_room(&port, &server_socket_fd);
    printf("Now port %u is activated and listenning\n", port);
    accept_connection(server_socket_fd, &client_socket_fd);
    bool offline = false;
    send_input(client_socket_fd, 2, 3, &offline, 0);
  } else {
    char* cur_name = info[1];
    unsigned short peer_port = atoi(info[2]);
    printf("peer_port read in is %d\n", peer_port);
    int socket_fd = 0;
    join_game_room(cur_name,peer_port, &socket_fd);
    int r;
    int c;
    bool offline;
    int result;
    get_opponent_input(socket_fd, &r, &c, &offline, &result);
  }

  return 0;
}
