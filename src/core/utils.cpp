#include "utils.hpp"

#include "errorhandling.hpp"
#include <cstring>
#include <iostream>

char *strcat_alloc(char *left, const char *right) {
  if (!left) {
    char *res = (char *)Utils::malloc(strlen(right) + 1);
    strncpy(res, right, strlen(right) + 1);
    return res;
  } else {
    size_t newlen = strlen(left) + strlen(right) + 1;
    char *res = Utils::realloc(left, newlen);
    strncat(res, right, newlen);
    return res;
  }
}
