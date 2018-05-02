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

/** \file  move_data_test.cpp Unit tests p4est_utils data transformation from
 *                            level to flat and vice-versa.
 *
*/

#include <boost/mpi.hpp>

#include <p8est.h>
#include <p8est_extended.h>
#include <p8est_connectivity.h>
#include <p8est_ghost.h>
#include <p8est_mesh.h>
#include <p8est_meshiter.h>
#include <p8est_virtual.h>

#include <iostream>
#include <vector>

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_MODULE move_data test
#define BOOST_TEST_ALTERNATIVE_INIT_API
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "p4est_utils.hpp"

typedef struct data{
  int foo;
  double bar;
} data_t;

bool operator==(data_t a, data_t b) {
  return a.foo == b.foo && a.bar == b.bar;
}

using boost::mpi::communicator;

BOOST_AUTO_TEST_CASE(move_data_check) {
  communicator comm;
  const int level = 4;

  auto verbosity = SC_LP_PRODUCTION;
  sc_init(comm, 1, 1, nullptr, verbosity);
  p4est_init(nullptr, verbosity);

  // initialize p4est structs
  p8est_connectivity_t * conn = p8est_connectivity_new_unitcube();
  p8est_t * p4est =
      p8est_new_ext(comm, conn, 0, level, 0, 0, nullptr, nullptr);
  p8est_balance(p4est, P8EST_CONNECT_FACE, nullptr);
  p8est_ghost_t * ghost = p8est_ghost_new(p4est, P8EST_CONNECT_FACE);
  p8est_mesh_t * mesh =
      p8est_mesh_new_ext(p4est, ghost, 1, 1, 1, P8EST_CONNECT_FACE);
  p8est_virtual_t * virt =
      p8est_virtual_new_ext(p4est, ghost, mesh, P8EST_CONNECT_FACE, 1);

  // allocate data
  std::vector< std::vector <data_t> > reference_data;
  std::vector< std::vector <data_t> > level_data;
  std::vector<data_t> flat_data(p4est->local_num_quadrants);
  p4est_utils_allocate_levelwise_storage(level_data, mesh, virt, true);
  p4est_utils_allocate_levelwise_storage(reference_data, mesh, virt, true);

  // populate data
  for (int lvl = 0; lvl < level; ++lvl) {
    p8est_meshiter_t *m =
        p8est_meshiter_new(p4est, ghost, mesh, virt, lvl, P8EST_CONNECT_FACE);
    int status = 0;
    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(m);
      if (status != P8EST_MESHITER_DONE) {
        data_t *c = &level_data[lvl][p8est_meshiter_get_current_storage_id(m)];
        c->foo = m->current_qid;
        c->bar = 0.11;
      }
    }
    p8est_meshiter_destroy(m);
  }

  // copy data into reference data
  for (int lvl = 0; lvl < level; ++lvl) {
    std::memcpy(reference_data[lvl].data(), level_data[lvl].data(),
                level_data[lvl].size() * sizeof(data_t));
  }

  // flatten data
  p4est_utils_flatten_data(p4est, mesh, virt, level_data, flat_data);
  p4est_utils_unflatten_data(p4est, mesh, virt, flat_data, level_data);

  for (int lvl = 0; lvl < level; ++lvl) {
    BOOST_CHECK(reference_data[lvl] == level_data[lvl]);
  }
  // cleanup
  p4est_utils_deallocate_levelwise_storage(level_data);
  p4est_utils_deallocate_levelwise_storage(reference_data);

  p8est_virtual_destroy(virt);
  p8est_mesh_destroy(mesh);
  p8est_ghost_destroy(ghost);
  p8est_destroy(p4est);
  p8est_connectivity_destroy(conn);
  sc_finalize();
}

int main(int argc, char **argv) {
  boost::mpi::environment mpi_env(argc, argv);

  return boost::unit_test::unit_test_main(init_unit_test, argc, argv);
}
