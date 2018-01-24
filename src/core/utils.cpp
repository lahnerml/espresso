#include "utils.hpp"

#include <cstring>
#include <iostream>

char *strcat_alloc(char *left, const char *right) {
  if (!left) {
    char *res = (char *)Utils::malloc(strlen(right) + 1);
    strcpy(res, right);
    return res;
  } else {
    char *res = (char *)Utils::realloc(left, strlen(left) + strlen(right) + 1);
    strcat(res, right);
    return res;
  }
}

int Utils::check_dangling_MPI_messages (MPI_Comm comm) {
  int flag;
  MPI_Status status;
  MPI_Barrier(comm);
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status);
  if (flag) {
    std::cout << "trailing message: found."
              << " source: " << status.MPI_SOURCE
              << " tag: " << status.MPI_TAG
              << " error code: " << status.MPI_ERROR
              << std::endl;
    errexit();
  }
  return 0;
}
