#include "utils.hpp"

#include "errorhandling.hpp"
#include <cstring>
#include <iostream>

char *strcat_alloc(char *left, const char *right) {
  if (!left) {
    return strdup(right);
  } else {
    size_t newlen = strlen(left) + strlen(right) + 1;
    char *res = Utils::realloc(left, newlen);
    strncat(res, right, newlen);
    return res;
  }
}
