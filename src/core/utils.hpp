/*
  Copyright (C) 2010-2018 The ESPResSo project
  Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
    Max-Planck-Institute for Polymer Research, Theory Group

  This file is part of ESPResSo.

  ESPResSo is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ESPResSo is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef UTILS_HPP
#define UTILS_HPP
/** \file utils.hpp
 *    Small functions that are useful not only for one modul.

 *  just some nice utilities...
 *  and some constants...
 *
*/

#include "Vector.hpp"
#include "utils/constants.hpp"
#include "utils/math/sqr.hpp"
#include "utils/memory.hpp"
#include "config.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace Utils {
  template<typename ForwardIterator, typename T>
  inline void iota_n(ForwardIterator first, size_t count, T value) {
    for (size_t i = 0; i < count; ++i) {
      *first = value + i;
      ++first;
    }
  }
}

namespace Utils {
/** For positive numbers, returns the log to basis two rounded down.
 * Returns 0 for negative numbers.
 */
template <typename T>
inline int nat_log2_floor(T x) {
  static_assert(std::is_integral<T>::value,
                "This version of logarithm can only be used for integer types\n");
  int lvl = 0;
  for (; x > 1; x >>= 1) lvl++;
  return lvl;
}
} // Namespace Utils

/*************************************************************/
/** \name Vector and matrix operations for three dimensons.  */
/*************************************************************/
/*@{*/

namespace Utils {
  /** verify that a double has at most log_10(div) trailing digits
   *
   * @param ai     integer representation of a * div
   * @param ad     double representation of a
   * @param div    measure the maximum allowed precision of ad
   */
  inline bool verify_prec(int ai, double ad, double div) {
    return (((ad - ROUND_ERROR_PREC) <= (ai / div)) &&
            ((ad + ROUND_ERROR_PREC) >= (ai / div)));
  }

  /** Calculate greatest common divisor of two integer-type variables
   *
   * @param [in] a, b  Integers to find gcd for
   */
  template <typename T>
  inline T gcd(T a, T b) {
    static_assert(std::is_integral<T>::value,
                  "GCD is only implemented for integer types\n");
    return b == 0 ? a : gcd(b, a % b);
  }

  /** Calculate greatest common divisor of arbitrary many variables
   *
   * @param [in] a, b   First two elements
   * @param [in] args   Remaining elements, may be empty
   */
  template <typename T, typename... Ts>
  inline T gcd(T a, T b, Ts... args) {
    return (gcd(gcd(a, b), args...));
  }

  /** Specialize greatest common divisor algorithm for doubles based using
   * \ref verify_prec
   *
   * @param [in] a, b  Doubles to find GCD for.
   */
  template <>
  inline double gcd<double>(double a, double b) {
    // allow at max 5 digits after .
    int factor = 10000;
    int aa = factor * a;
    int bb = factor * b;
    if (!verify_prec(aa, a, factor)) {
      fprintf(stderr, "input is not representable as int(d) + d-int(d)/%i. "
              "(%lf)\n", factor, a);
      return -1.0;
    }
    if (!verify_prec(bb, b, factor)) {
      fprintf(stderr, "input is not representable as int(d) + d-int(d)/%i. "
              "(%lf)\n", factor, b);
      return -1.0;
    }

    // reduce aa/factor and bb/factor
    int gcd_a = gcd(aa, factor);
    int gcd_b = gcd(bb, factor);

    // calculate shares of a and b to a naive common denominator
    int common_denom_a = factor / gcd_a;
    int common_denom_b = factor / gcd_b;

    // calculate greatest common denominiator for numerator and denominator
    int gcd_denom =
        gcd(common_denom_b * factor / gcd_a, common_denom_a * factor / gcd_b);
    int gcd_num = gcd(common_denom_b * aa / gcd_a, common_denom_a * bb / gcd_b);

    return (double)gcd_num / (double)gcd_denom;
  }
}
/*@}*/

/** calculates the scalar product of two vectors a nd b */
template <typename T1, typename T2> double scalar(const T1 &a, const T2 &b) {
  double d2 = 0.0;
  for (int i = 0; i < 3; i++)
    d2 += a[i] * b[i];
  return d2;
}

/** calculates the squared length of a vector */
template <typename T> double sqrlen(T const &v) { return scalar(v, v); }

/** calculates the length of a vector */
inline double normr(double v[3]) { return std::sqrt(sqrlen(v)); }

/** calculates unit vector */
inline void unit_vector(double v[3], double y[3]) {
  const double d = normr(v);

  for (int i = 0; i < 3; i++)
    y[i] = v[i] / d;
}

/** calculates the vector product c of two vectors a and b */
template <typename T, typename U, typename V>
inline void vector_product(T const &a, U const &b, V &c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
  return;
}

/*@}*/

/*************************************************************/
/** \name Three dimensional grid operations                  */
/*************************************************************/
/*@{*/

/** get the linear index from the position (a,b,c) in a 3D grid
 *  of dimensions adim[]. returns linear index.
 *
 * @return        the linear index
 * @param a       x position
 * @param b       y position
 * @param c       z position
 * @param adim    dimensions of the underlying grid
 */
inline int get_linear_index(int a, int b, int c, const Vector3i &adim) {
  assert((a >= 0) && (a < adim[0]));
  assert((b >= 0) && (b < adim[1]));
  assert((c >= 0) && (c < adim[2]));

  return (a + adim[0] * (b + adim[1] * c));
}

/** get the position a[] from the linear index in a 3D grid
 *  of dimensions adim[].
 *
 * @param i       linear index
 * @param a       x position (return value)
 * @param b       y position (return value)
 * @param c       z position (return value)
 * @param adim    dimensions of the underlying grid
 */
inline void get_grid_pos(int i, int *a, int *b, int *c, const Vector3i &adim) {
  *a = i % adim[0];
  i /= adim[0];
  *b = i % adim[1];
  i /= adim[1];
  *c = i;
}

/*@}*/

/*************************************************************/
/** \name Distance calculations.  */
/*************************************************************/
/*@{*/

/** returns the distance between two position.
 *  \param pos1 Position one.
 *  \param pos2 Position two.
 */
inline double distance2(const Vector3d &a, const Vector3d &b) {
  return (a - b).norm2();
}

/*@}*/

#endif
