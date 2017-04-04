#include <cstring>
#include "utils.hpp"

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


//--------------------------------------------------------------------------------------------------
// Returns the morton index for given cartesian coordinates.
// Note: This is not the index of the p4est quadrants. But the ordering is the same.
int64_t dd_p4est_cell_morton_idx(int x, int y, int z) {
  //p4est_quadrant_t c;
  //c.x = x; c.y = y; c.z = z;
  /*if (x < 0 || x >= grid_size[0])
    runtimeErrorMsg() << x << "x" << y << "x" << z << " no valid cell";
  if (y < 0 || y >= grid_size[1])
    runtimeErrorMsg() << x << "x" << y << "x" << z << " no valid cell";
  if (z < 0 || z >= grid_size[2])
    runtimeErrorMsg() << x << "x" << y << "x" << z << " no valid cell";*/
  
  int64_t idx = 0;
  int64_t pos = 1;
  
  for (int i = 0; i < 21; ++i) {
    if ((x&1)) idx += pos;
    x >>= 1; pos <<= 1;
    if ((y&1)) idx += pos;
    y >>= 1; pos <<= 1;
    if ((z&1)) idx += pos;
    z >>= 1; pos <<= 1;
  }
  
  return idx;
  //c.level = P4EST_QMAXLEVEL;
  //return p4est_quadrant_linear_id(&c,P4EST_QMAXLEVEL);
}
