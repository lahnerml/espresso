/*
  Copyright (C) 2017 The ESPResSo project

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

/** \file  Morton_test.cpp Unit tests p4est_utils Morton index calculation
 *
*/

#define BOOST_TEST_MODULE morton test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "utils/Morton.hpp"
#include <array>
#include <random>

BOOST_AUTO_TEST_CASE(Morton_Index_Check) {
  const int n_experiments = 10000;

  const int dim = 3;
  const int lvl = 21;
  const int64_t n_idx = static_cast<uint64_t>(1) << (dim * lvl);

  std::default_random_engine gen;
  std::uniform_int_distribution<uint64_t> distribution(0, n_idx);

  std::array<uint64_t, 3> tmp;
  uint64_t sample;
  for (int idx = 0; idx < n_experiments; ++idx) {
    sample = distribution(gen);
    tmp = Utils::morton_idx_to_coords(sample);
    BOOST_CHECK(sample == Utils::morton_coords_to_idx(tmp[0], tmp[1], tmp[2]));
  }
}
