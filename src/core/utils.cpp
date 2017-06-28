#include <cstring>
#include "utils.hpp"

#ifdef __BMI2__
#include <x86intrin.h>
#endif

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
#ifdef __BMI2__
#warning "Using BMI2 for cell_morton_idx"
  static const unsigned mask_x = 0x49249249;
  static const unsigned mask_y = 0x92492492;
  static const unsigned mask_z = 0x24924924;

  return _pdep_u32(x, mask_x)
           | _pdep_u32(y, mask_y)
           | _pdep_u32(z, mask_z);
#else
#warning "BMI2 not detected: Using slow loop version for cell_morton_idx"
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
#endif
}
