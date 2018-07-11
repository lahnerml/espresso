#include <array>
#include <cstdint>

// For intrinsics version of cell_morton_idx
#ifdef __BMI2__
#include <x86intrin.h>
#endif

namespace Utils {
#ifdef __BMI2__
inline unsigned _d2x(unsigned d, unsigned mask)
{
  return _pext_u64(d, mask);
}
#endif

/** Map a virtual Morton-index to a grid_count on a regular grid
 *
 * @param idx   A Morton-index calculated by interleaving a cell count
 * @return      The de-interleaved coordinates
 */
inline std::array<uint64_t, 3> morton_idx_to_coords (uint64_t idx) {
#ifdef __BMI2__
  static const uint64_t mask_x = 0x1249249249249249;
  static const uint64_t mask_y = 0x2492492492492492;
  static const uint64_t mask_z = 0x4924924924924924;

  std::array<uint64_t, 3> res = {{
      _pext_u64(idx, mask_x),
      _pext_u64(idx, mask_y),
      _pext_u64(idx, mask_z)
  }};
#else
  uint64_t x = 0;
  uint64_t y = 0;
  uint64_t z = 0;
  for (uint64_t i = 0; i < 21; ++i) {
    // When doing this by hand, i resembles the level. We inspect if the bit for
    // the respective direction is set on the current level and if it is, we
    // also set it in the result array.
    x |= ((idx & (static_cast<uint64_t>(1) << (3 * i + 0))) != 0 ? 1 : 0) << i;
    y |= ((idx & (static_cast<uint64_t>(1) << (3 * i + 1))) != 0 ? 1 : 0) << i;
    z |= ((idx & (static_cast<uint64_t>(1) << (3 * i + 2))) != 0 ? 1 : 0) << i;
  }
  std::array<uint64_t, 3> res = {{x, y, z}};
#endif
  return res;
}

/** Interleave three coordinates to a (virtual) Morton index.
 *
 * @param xyz  Spatial position
 * @return     Morton index
 */
inline int64_t morton_coords_to_idx(int x, int y, int z) {
#ifdef __BMI2__
  //#warning "Using BMI2 for cell_morton_idx"
  static const uint64_t mask_x = 0x1249249249249249;
  static const uint64_t mask_y = 0x2492492492492492;
  static const uint64_t mask_z = 0x4924924924924924;;

  return _pdep_u64(x, mask_x)
           | _pdep_u64(y, mask_y)
           | _pdep_u64(z, mask_z);
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
