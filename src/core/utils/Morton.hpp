#include <array>

// For intrinsics version of cell_morton_idx
#ifdef __BMI2__
#include <x86intrin.h>
#endif

namespace Utils {
#ifdef __BMI2__
inline unsigned _d2x(unsigned d, unsigned mask)
{
  return _pext_u32(d, mask);
}
#endif

inline std::array<unsigned int, 3> morton_idx_to_coords (int64_t idx) {
#ifdef __BMI2__
  static const unsigned mask_x = 0x49249249;
  static const unsigned mask_y = 0x92492492;
  static const unsigned mask_z = 0x24924924;

  std::array<unsigned int, 3> res = {{
      _d2x(idx, mask_x),
      _d2x(idx, mask_y),
      _d2x(idx, mask_z)
  }};
#else
  unsigned int x = 0;
  unsigned int y = 0;
  unsigned int z = 0;
  for (int i = 0; i < 21; ++i) {
    x |= ((idx & (1 << (3 * i + 0))) != 0 ? 1 : 0) << i;
    y |= ((idx & (1 << (3 * i + 1))) != 0 ? 1 : 0) << i;
    z |= ((idx & (1 << (3 * i + 2))) != 0 ? 1 : 0) << i;
  }
  std::array<unsigned int, 3> res = {{x, y, z}};
#endif
  return res;
}

inline int64_t morton_coords_to_idx(int x, int y, int z) {
#ifdef __BMI2__
  //#warning "Using BMI2 for cell_morton_idx"
  static const unsigned mask_x = 0x49249249;
  static const unsigned mask_y = 0x92492492;
  static const unsigned mask_z = 0x24924924;

  return _pdep_u32(x, mask_x)
           | _pdep_u32(y, mask_y)
           | _pdep_u32(z, mask_z);
#else
//#warning "BMI2 not detected: Using slow loop version for cell_morton_idx"
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
}
