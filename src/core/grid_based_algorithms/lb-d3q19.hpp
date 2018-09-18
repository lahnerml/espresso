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
/** \file lb-d3q19.hpp
 * Header file for the lattice Boltzmann D3Q19 model.
 *
 * This header file contains the definition of the D3Q19 model.
 */

#ifndef D3Q19_H
#define D3Q19_H

#ifdef LB
#ifdef LB_ADAPTIVE_GPU
#include "grid_based_algorithms/lb-adaptive-gpu.hpp"
#else // LB_ADAPTIVE_GPU
#include "grid_based_algorithms/lb.hpp"
#endif // LB_ADAPTIVE_GPU

/** Velocity sub-lattice of the D3Q19 model */
// clang-format off
#ifndef LB_ADAPTIVE
static constexpr const std::array<std::array<double, 3>, 19> d3q19_lattice = {
#else // LB_ADAPTIVE
static constexpr const std::array<std::array<lb_float, 3>, 19> d3q19_lattice = {
#endif // LB_ADAPTIVE
  {{{  0.,  0.,  0. }},
   {{  1.,  0.,  0. }},
   {{ -1.,  0.,  0. }},
   {{  0.,  1.,  0. }},
   {{  0., -1.,  0. }},
   {{  0.,  0.,  1. }},
   {{  0.,  0., -1. }},
   {{  1.,  1.,  0. }},
   {{ -1., -1.,  0. }},
   {{  1., -1.,  0. }},
   {{ -1.,  1.,  0. }},
   {{  1.,  0.,  1. }},
   {{ -1.,  0., -1. }},
   {{  1.,  0., -1. }},
   {{ -1.,  0.,  1. }},
   {{  0.,  1.,  1. }},
   {{  0., -1., -1. }},
   {{  0.,  1., -1. }},
   {{  0., -1.,  1. }}}};
// clang-format on

/** Coefficients for pseudo-equilibrium distribution of the D3Q19 model */
// clang-format off
#ifndef LB_ADAPTIVE
static constexpr const std::array<std::array<double, 4>, 19>
#else // LB_ADAPTIVE
static constexpr const std::array<std::array<lb_float, 4>, 19>
#endif // LB_ADAPTIVE
  d3q19_coefficients = {{{{1. /  3., 1.      , 3. / 2., -1. /  2.}},
                         {{1. / 18., 1. /  6., 1. / 4., -1. / 12.}},
                         {{1. / 18., 1. /  6., 1. / 4., -1. / 12.}},
                         {{1. / 18., 1. /  6., 1. / 4., -1. / 12.}},
                         {{1. / 18., 1. /  6., 1. / 4., -1. / 12.}},
                         {{1. / 18., 1. /  6., 1. / 4., -1. / 12.}},
                         {{1. / 18., 1. /  6., 1. / 4., -1. / 12.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}},
                         {{1. / 36., 1. / 12., 1. / 8., -1. / 24.}}}};
// clang-format on

/** Coefficients in the functional for the equilibrium distribution */
#ifndef LB_ADAPTIVE
static constexpr const std::array<double, 19> d3q19_w = {
#else  // LB_ADAPTIVE
static constexpr const std::array<lb_float, 19> d3q19_w = {
#endif // LB_ADAPTIVE
    {1. / 3.,  1. / 18., 1. / 18., 1. / 18., 1. / 18., 1. / 18., 1. / 18.,
     1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36.,
     1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36.}};

/** Basis of the mode space as described in [Duenweg, Schiller, Ladd] */
#ifndef LB_ADAPTIVE
static constexpr const std::array<std::array<double, 19>, 20> d3q19_modebase = {
#else  // LB_ADAPTIVE
static constexpr const std::array<std::array<lb_float, 19>, 20> d3q19_modebase = {
#endif // LB_ADAPTIVE
     {{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0}},
      {{0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        -1.0, 0.0, 0.0, 0.0, 0.0}},
     {{0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, -1.0, 1.0, -1.0}},
     {{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0,
        1.0, 1.0, -1.0, -1.0, 1.0}},
     {{-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0}},
     {{0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
       1.0, -1.0, -1.0, -1.0, -1.0}},
     {{-0.0, 1.0, 1.0, 1.0, 1.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0,
       -1.0, -1.0, -1.0, -1.0, -1.0, -1.0}},
     {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0}},
     {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0,
       -1.0, 0.0, 0.0, 0.0, 0.0}},
     {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       1.0, 1.0, -1.0, -1.0}},
     {{0.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
      -1.0, 0.0, 0.0, 0.0, 0.0}},
     {{0.0, 0.0, 0.0, -2.0, 2.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0,
       0.0, 1.0, -1.0, 1.0, -1.0}},
     {{0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0,
       1.0, 1.0, -1.0, -1.0, 1.0}},
     {{0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
       1.0, 0.0, 0.0, 0.0, 0.0}},
     {{0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0,
       0.0, -1.0, 1.0, -1.0, 1.0}},
     {{0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0,
       1.0, -1.0, 1.0, 1.0, -1.0}},
     {{1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
     {{0.0, -1.0, -1.0, 1.0, 1.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
       1.0, -1.0, -1.0, -1.0, -1.0}},
     {{0.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0,
       -1.0, -1.0, -1.0, -1.0, -1.0, -1.0}},
     /* the following values are the (weighted) lengths of the vectors */
     {{1.0, 1. / 3., 1. / 3., 1. / 3., 2. / 3., 4. / 9., 4. / 3., 1. / 9.,
       1. / 9., 1. / 9., 2. / 3., 2. / 3., 2. / 3., 2. / 9., 2. / 9., 2. / 9.,
       2.0, 4. / 9., 4. / 3.}}}};

#endif /* LB */

#endif /* D3Q19_H */

/*@}*/
