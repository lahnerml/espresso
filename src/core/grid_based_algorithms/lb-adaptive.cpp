/*
   Copyright (C) 2010,2011,2012,2013,2014,2015 The ESPResSo project
   Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
   Max-Planck-Institute for Polymer Research, Theory Group,

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

/** \file lb-adaptive.cpp
 *
 * Adaptive Lattice Boltzmann Scheme using CPU.
 * Implementation file for \ref lb-adaptive.hpp.
 */
#include "grid_based_algorithms/lb-adaptive.hpp"
#include "communication.hpp"
#include "grid_based_algorithms/lb-adaptive-gpu.hpp"
#include "grid_based_algorithms/lb-d3q19.hpp"
#include "grid_based_algorithms/lb.hpp"
#include "grid_based_algorithms/lbboundaries.hpp"
#include "p4est_utils.hpp"
#include "random.hpp"
#include "thermostat.hpp"
#include "utils.hpp"
#include "utils/Morton.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

#ifdef LB_ADAPTIVE

#define DUMP_VIRTUALS

/* Code duplication from lb.cpp */
/* For the D3Q19 model most functions have a separate implementation
 * where the coefficients and the velocity vectors are hardcoded
 * explicitly. This saves a lot of multiplications with 1's and 0's
 * thus making the code more efficient. */

#if (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) && !defined(GAUSSRANDOM))
#define FLATNOISE
#endif // (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) &&
       // !defined(GAUSSRANDOM))

/* "external variables" */
std::vector<std::vector<lbadapt_payload_t>> lbadapt_local_data;
std::vector<std::vector<lbadapt_payload_t>> lbadapt_ghost_data;

/*** MAPPING OF CI FROM ESPRESSO LBM TO P4EST FACE-/EDGE ENUMERATION ***/
/**
 * | ESPResSo c_i | p4est face | p4est edge | vec          |
 * |--------------+------------+------------+--------------|
 * |            0 |          - |          - | { 0,  0,  0} |
 * |            1 |          1 |          - | { 1,  0,  0} |
 * |            2 |          0 |          - | {-1,  0,  0} |
 * |            3 |          3 |          - | { 0,  1,  0} |
 * |            4 |          2 |          - | { 0, -1,  0} |
 * |            5 |          5 |          - | { 0,  0,  1} |
 * |            6 |          4 |          - | { 0,  0, -1} |
 * |            7 |          - |         11 | { 1,  1,  0} |
 * |            8 |          - |          8 | {-1, -1,  0} |
 * |            9 |          - |          9 | { 1, -1,  0} |
 * |           10 |          - |         10 | {-1,  1,  0} |
 * |           11 |          - |          7 | { 1,  0,  1} |
 * |           12 |          - |          4 | {-1,  0, -1} |
 * |           13 |          - |          5 | { 1,  0, -1} |
 * |           14 |          - |          6 | {-1,  0,  1} |
 * |           15 |          - |          3 | { 0,  1,  1} |
 * |           16 |          - |          0 | { 0, -1, -1} |
 * |           17 |          - |          1 | { 0,  1, -1} |
 * |           18 |          - |          2 | { 0, -1,  1} |
 */

/*** SETUP ***/
void lbadapt_allocate_data() {
  p4est_utils_allocate_levelwise_storage(lbadapt_local_data, adapt_mesh,
                                         adapt_virtual, true);

  /** ghost */
  if (adapt_ghost->ghosts.elem_count == 0) {
    return;
  } else {
    p4est_utils_allocate_levelwise_storage(lbadapt_ghost_data, adapt_mesh,
                                           adapt_virtual, false);
  }

#ifdef LB_ADAPTIVE_GPU
  local_num_quadrants = adapt_mesh->local_num_quadrants;
  lbadapt_gpu_allocate_device_memory();
#endif // LB_ADAPTIVE_GPU
} // lbadapt_allocate_data();

void lbadapt_release() {
  /** cleanup custom managed payload */
  p4est_utils_deallocate_levelwise_storage(lbadapt_local_data);
  p4est_utils_deallocate_levelwise_storage(lbadapt_ghost_data);
#ifdef LB_ADAPTIVE_GPU
  lbadapt_gpu_deallocate_device_memory();
#endif // LB_ADAPTIVE_GPU
}

#ifndef LB_ADAPTIVE_GPU
void init_to_zero(lbadapt_payload_t *data) {
#else  // LB_ADAPTIVE_GPU
void init_to_zero(lbadapt_patch_cell_t *data) {
#endif // LB_ADAPTIVE_GPU
  for (int i = 0; i < LB_Model<>::n_veloc; i++) {
    data->lbfluid[0][i] = 0.;
    data->lbfluid[1][i] = 0.;
  }

#ifndef LB_ADAPTIVE_GPU
  // ints
  data->lbfields.recalc_fields = 1;
  data->lbfields.has_force_density = 0;

  // 1D "array"
  data->lbfields.rho[0] = 0.;

  // 3D arrays
  for (int i = 0; i < 3; i++) {
    data->lbfields.j[i] = 0;
    data->lbfields.force_density[i] = 0;
#ifdef IMMERSED_BOUNDARY
    data->lbfields.force_buf[i] = 0;
#endif // IMMERSED_BOUNDARY
  }

  // 6D array
  for (double &i : data->lbfields.pi) {
    i = 0;
  }
#endif // LB_ADAPTIVE_GPU
}

#ifndef LB_ADAPTIVE_GPU
void lbadapt_set_force(lbadapt_payload_t *data, int level)
#else  // LB_ADAPTIVE_GPU
void lbadapt_set_force(lbadapt_patch_cell_t *data, int level)
#endif // LB_ADAPTIVE_GPU
{
  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];
#ifdef LB_ADAPTIVE_GPU
  // unit conversion: force density
  data->force[0] = prefactors[level] * lbpar.ext_force_density[0] *
                   Utils::sqr(h_max) * Utils::sqr(lbpar.tau);
  data->force[1] = prefactors[level] * lbpar.ext_force_density[1] *
                   Utils::sqr(h_max) * Utils::sqr(lbpar.tau);
  data->force[2] = prefactors[level] * lbpar.ext_force_density[2] *
                   Utils::sqr(h_max) * Utils::sqr(lbpar.tau);
#else  // LB_ADAPTIVE_GPU
  data->lbfields.force_density[0] = p4est_params.prefactors[level] *
                                    lbpar.ext_force_density[0] *
                                    Utils::sqr(h_max) * Utils::sqr(lbpar.tau);
  data->lbfields.force_density[1] = p4est_params.prefactors[level] *
                                    lbpar.ext_force_density[1] *
                                    Utils::sqr(h_max) * Utils::sqr(lbpar.tau);
  data->lbfields.force_density[2] = p4est_params.prefactors[level] *
                                    lbpar.ext_force_density[2] *
                                    Utils::sqr(h_max) * Utils::sqr(lbpar.tau);
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_init() {
  // reset p4est
  mpi_lbadapt_grid_init(0, 0);

  // reset data
  lbadapt_local_data.clear();
  lbadapt_ghost_data.clear();
#ifdef LB_ADAPTIVE_GPU
  lbadapt_gpu_deallocate_device_memory();
#endif // LB_ADAPTIVE_GPU
  lbadapt_allocate_data();

  int status;
  lbadapt_payload_t *data;
  castable_unique_ptr<p8est_meshiter_t> mesh_iter;
  for (int level = 0; level < P8EST_MAXLEVEL; ++level) {
    status = 0;
    mesh_iter.reset(p8est_meshiter_new_ext(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
        P8EST_CONNECT_EDGE, P8EST_TRAVERSE_LOCALGHOST,
        P8EST_TRAVERSE_REALVIRTUAL, P8EST_TRAVERSE_PARBOUNDINNER));

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        if (!mesh_iter->current_is_ghost) {
          data = &lbadapt_local_data[level].at(
              p8est_meshiter_get_current_storage_id(mesh_iter));
        } else {
          data = &lbadapt_ghost_data[level].at(
              p8est_meshiter_get_current_storage_id(mesh_iter));
        }
#ifndef LB_ADAPTIVE_GPU
        data->lbfields.boundary = 0;
        init_to_zero(data);
#else  // LB_ADAPTIVE_GPU
        for (int patch_z = 0; patch_z < LBADAPT_PATCHSIZE_HALO; ++patch_z) {
          for (int patch_y = 0; patch_y < LBADAPT_PATCHSIZE_HALO; ++patch_y) {
            for (int patch_x = 0; patch_x < LBADAPT_PATCHSIZE_HALO; ++patch_x) {
              init_to_zero(&data->patch[patch_x][patch_y][patch_z]);
              data->patch[patch_x][patch_y][patch_z].boundary = 0;
            }
          }
        }
#endif // LB_ADAPTIVE_GPU
      }
    }
  }
} // lbadapt_init();

void lbadapt_reinit_parameters() {
  for (int i = p4est_params.max_ref_level; p4est_params.min_ref_level <= i;
       --i) {
    p4est_params.prefactors[i] = 1 << (p4est_params.max_ref_level - i);

#ifdef LB_ADAPTIVE_GPU
    p4est_params.h[i] = ((double)P8EST_QUADRANT_LEN(i) * box_l[0]) /
                        ((double)LBADAPT_PATCHSIZE * (double)P8EST_ROOT_LEN *
                         (double)lb_conn_brick[0]);
#else  // LB_ADAPTIVE_GPU
    p4est_params.h[i] = ((double)P8EST_QUADRANT_LEN(i) * box_l[0]) /
                        ((double)P8EST_ROOT_LEN * (double)lb_conn_brick[0]);
#endif // LB_ADAPTIVE_GPU
#ifndef USE_BGK
    if (lbpar.viscosity > 0.0) {
      lbpar.gamma_shear[i] =
          1. - 2. / (6. * lbpar.viscosity * p4est_params.prefactors[i] *
                         lbpar.tau / (Utils::sqr(p4est_params.h[i])) +
                     1.);
    }
    if (lbpar.bulk_viscosity > 0.0) {
      lbpar.gamma_bulk[i] =
          1. - 2. / (9. * lbpar.bulk_viscosity * lbpar.tau /
                         (p4est_params.prefactors[i] *
                          Utils::sqr(p4est_params.h[i])) +
                     1.);
    }
#else
    lbpar.gamma_shear[i] = 0.0; // uncomment for special case of BGK
    lbpar.gamma_bulk[i] = 0.0;
#endif
  }
  lbpar.gamma_odd = 0.0;
  lbpar.gamma_even = 0.0;
#ifdef LB_ADAPTIVE_GPU
  memcpy(lbpar.prefactors, prefactors, P8EST_MAXLEVEL * sizeof(int));
  memcpy(lbpar.h, h, P8EST_MAXLEVEL * sizeof(lb_float));
  memcpy(lbpar.gamma_bulk, gamma_bulk, P8EST_MAXLEVEL * sizeof(lb_float));
  memcpy(lbpar.gamma_shear, gamma_shear, P8EST_MAXLEVEL * sizeof(lb_float));

  lbadapt_gpu_init();
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_reinit_force_per_cell() {
  if (lbadapt_local_data.empty()) {
    lbadapt_allocate_data();
  }
  int status;
  lbadapt_payload_t *data;
  castable_unique_ptr<p4est_meshiter_t> mesh_iter;
  for (int level = 0; level < P8EST_MAXLEVEL; ++level) {
    status = 0;

    mesh_iter.reset(p8est_meshiter_new_ext(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
        P8EST_CONNECT_EDGE, P8EST_TRAVERSE_LOCALGHOST, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER));

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        if (!mesh_iter->current_is_ghost) {
          data = &lbadapt_local_data[level].at(
              p8est_meshiter_get_current_storage_id(mesh_iter));
        } else {
          data = &lbadapt_ghost_data[level].at(
              p8est_meshiter_get_current_storage_id(mesh_iter));
        }
#ifndef LB_ADAPTIVE_GPU
        lbadapt_set_force(data, level);
#else  // LB_ADAPTIVE_GPU
        for (int patch_z = 1; patch_z <= LBADAPT_PATCHSIZE; ++patch_z) {
          for (int patch_y = 1; patch_y <= LBADAPT_PATCHSIZE; ++patch_y) {
            for (int patch_x = 1; patch_x <= LBADAPT_PATCHSIZE; ++patch_x) {
              lbadapt_set_force(&data->patch[patch_x][patch_y][patch_z], level);
            }
          }
        }
#endif // LB_ADAPTIVE_GPU
      }
    }
  }
}

void lbadapt_reinit_fluid_per_cell() {
  if (adapt_p4est == nullptr) {
    lbadapt_init();
  }
  lbadapt_release();
  lbadapt_allocate_data();

  int status;
  lbadapt_payload_t *data;
  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  /** prepare exchanging boundary values */
  std::vector<lbadapt_payload_t *> local_pointer(P8EST_QMAXLEVEL);
  std::vector<lbadapt_payload_t *> ghost_pointer(P8EST_QMAXLEVEL);
  prepare_ghost_exchange(lbadapt_local_data, local_pointer, lbadapt_ghost_data,
                         ghost_pointer);
  castable_unique_ptr<p4est_meshiter_t> mesh_iter;
  for (int level = p4est_params.min_ref_level;
       level <= p4est_params.max_ref_level; ++level) {
    status = 0;

    mesh_iter.reset(p8est_meshiter_new_ext(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
        P8EST_CONNECT_EDGE, P8EST_TRAVERSE_LOCALGHOST, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER));

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        if (!mesh_iter->current_is_ghost) {
          data = &lbadapt_local_data[level].at(
              p8est_meshiter_get_current_storage_id(mesh_iter));
        } else {
          data = &lbadapt_ghost_data[level].at(
              p8est_meshiter_get_current_storage_id(mesh_iter));
        }
        // convert rho to lattice units
        lb_float rho = lbpar.rho * h_max * h_max * h_max;
        // start with fluid at rest and no stress
        lb_float j[3] = {0., 0., 0.};
        std::array<lb_float, 6> pi = {{0., 0., 0., 0., 0., 0.}};
#ifndef LB_ADAPTIVE_GPU
        lbadapt_calc_n_from_rho_j_pi(data->lbfluid, rho, j, pi,
                                     p4est_params.h[level]);
#ifdef LB_BOUNDARIES
        if (!mesh_iter->current_is_ghost) {
          double mp[3];
          p4est_utils_get_midpoint(mesh_iter, mp);
          data->lbfields.boundary = lbadapt_is_boundary(mp);
        }
#endif // LB_BOUNDARIES
#else  // LB_ADAPTIVE_GPU
        for (int patch_z = 1; patch_z <= LBADAPT_PATCHSIZE; ++patch_z) {
          for (int patch_y = 1; patch_y <= LBADAPT_PATCHSIZE; ++patch_y) {
            for (int patch_x = 1; patch_x <= LBADAPT_PATCHSIZE; ++patch_x) {
              lbadapt_calc_n_from_rho_j_pi(
                  data->patch[patch_x][patch_y][patch_z].lbfluid, rho, j, pi,
                  h[level]);
              data->patch[patch_x][patch_y][patch_z].boundary = 0;
            }
          }
        }
#endif // LB_ADAPTIVE_GPU
      }
    }
    p4est_virtual_ghost_exchange_data_level(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual,
        adapt_virtual_ghost, level, sizeof(lbadapt_payload_t),
        (void **)local_pointer.data(), (void **)ghost_pointer.data());
  }
}

template <>
int data_transfer<lbadapt_payload_t>(p8est_t *p4est_old, p8est_t *p4est_new,
                                     p8est_quadrant_t *quad_old,
                                     p8est_quadrant_t *quad_new, int which_tree,
                                     lbadapt_payload_t *data_old,
                                     lbadapt_payload_t *data_new) {
#ifndef LB_ADAPTIVE_GPU
  // FIXME Port to GPU
  std::memcpy(data_new, data_old, sizeof(lbadapt_payload_t));
#endif // LB_ADAPTIVE_GPU
  return 0;
}

template <>
int data_restriction<lbadapt_payload_t>(p8est_t *p4est_old, p8est_t *p4est_new,
                                        p8est_quadrant_t *quad_old,
                                        p8est_quadrant_t *quad_new,
                                        int which_tree,
                                        lbadapt_payload_t *data_old,
                                        lbadapt_payload_t *data_new) {
#ifndef LB_ADAPTIVE_GPU
  // FIXME Port to GPU
  // verify that level is correct
  P4EST_ASSERT(quad_new->level + 1 == quad_old->level);

  // check boundary status.
  std::array<double, 3> new_mp{};
  int new_boundary;
  p4est_utils_get_midpoint(p4est_new, which_tree, quad_new, new_mp.data());

  new_boundary = lbadapt_is_boundary(new_mp.data());

  if (new_boundary) {
    // if cell is a boundary cell initialize data to 0.
    init_to_zero(data_new);
    data_new->lbfields.boundary = new_boundary;
  } else {
    // else calculate arithmetic mean of population of smaller quadrants
    for (int vel = 0; vel < LB_Model<>::n_veloc; ++vel) {
      data_new->lbfluid[0][vel] += 0.125 * data_old->lbfluid[0][vel];
    }
  }

  // finally re-initialize local force
  // TODO: Can this be optimized? E.g. by has_force_density flag?
  lbadapt_set_force(data_new, quad_new->level);

#endif // !LB_ADAPTIVE_GPU
  return 0;
}

template <>
int data_interpolation<lbadapt_payload_t>(
    p8est_t *p4est_old, p8est_t *p4est_new, p8est_quadrant_t *quad_old,
    p8est_quadrant_t *quad_new, int which_tree, lbadapt_payload_t *data_old,
    lbadapt_payload_t *data_new) {
#ifndef LB_ADAPTIVE_GPU
  // verify that level is correct
  P4EST_ASSERT(quad_new->level == 1 + quad_old->level);

  // check boundary status.
  std::array<double, 3> new_mp{};
  int new_boundary;
  p4est_utils_get_midpoint(p4est_new, which_tree, quad_new, new_mp.data());

  new_boundary = lbadapt_is_boundary(new_mp.data());

  if (new_boundary) {
    // if cell is a boundary cell initialize data to 0.
    init_to_zero(data_new);
    data_new->lbfields.boundary = new_boundary;
  } else {
    std::memcpy(data_new, data_old, sizeof(lbadapt_payload_t));
  }
  // re-initialize force
  lbadapt_set_force(data_new, quad_new->level);

#endif // !LB_ADAPTIVE_GPU
  return 0;
}

int lbadapt_is_boundary(double *pos) {
  double dist, dist_tmp;
  std::array<double, 3> dist_vec{};
  dist = std::numeric_limits<double>::max();
  int the_boundary = -1;
  int n = 0;

  for (auto it = LBBoundaries::lbboundaries.begin();
       it != LBBoundaries::lbboundaries.end(); ++it, ++n) {
    (**it).calc_dist(pos, &dist_tmp, dist_vec.data());

    if (dist_tmp < dist) {
      dist = dist_tmp;
      the_boundary = n;
    }
  }

  if (dist <= 0 && !LBBoundaries::lbboundaries.empty()) {
    return the_boundary + 1;
  } else {
    return 0;
  }
}

#ifdef LB_ADAPTIVE_GPU
void lbadapt_patches_populate_halos(int level) {
  // clang-format off
    const int inv[] = { 0,
                        2,  1,  4,  3,  6,  5,
                        8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17 };
  // clang-format on
  lbadapt_payload_t *data, *neighbor_data;
  int status = 0;
  castable_unique_ptr<p8est_meshiter_t> mesh_iter = p8est_meshiter_new_ext(
      adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
      P8EST_CONNECT_EDGE, P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      if (!mesh_iter->current_is_ghost) {
        data = &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
            mesh_iter)];
      } else {
        SC_ABORT_NOT_REACHED();
      }
      for (int dir_ESPR = 1; dir_ESPR < 19; ++dir_ESPR) {
        // convert direction
        int dir_p4est = ci_to_p4est[(dir_ESPR - 1)];
        // set neighboring cell information in iterator
        p8est_meshiter_set_neighbor_quad_info(mesh_iter, dir_p4est);

        if (mesh_iter->neighbor_qid != -1) {
          int inv_neigh_dir_p4est = mesh_iter->neighbor_entity_index;
          int inv_neigh_dir_ESPR = p4est_to_ci[inv_neigh_dir_p4est];

          assert(inv[dir_ESPR] == inv_neigh_dir_ESPR);
          assert(dir_ESPR == inv[inv_neigh_dir_ESPR]);

          if (mesh_iter->neighbor_is_ghost) {
            neighbor_data =
                &lbadapt_ghost_data[level]
                                   [p8est_meshiter_get_neighbor_storage_id(
                                       mesh_iter)];
          } else {
            neighbor_data =
                &lbadapt_local_data[level]
                                   [p8est_meshiter_get_neighbor_storage_id(
                                       mesh_iter)];
          }

          // before reading or writing 2 tasks need to be performed:
          // a) set basic offsets for reading and writing data
          // b) decide for each direction the number of iterations that needs to
          //    be performed (1 or |cells per patch|)
          int r_offset_x, r_offset_y, r_offset_z;
          int w_offset_x, w_offset_y, w_offset_z;
          int iter_max_x, iter_max_y, iter_max_z;

          if (0 <= dir_p4est && dir_p4est < P8EST_FACES) {
            // for faces:
            // The face is orthogonal to the direction it is associated with.
            // That means for populating the halo of the patch we have to
            // iterate over the other two indices, keeping the original
            // direction constant.
            iter_max_x = iter_max_y = iter_max_z = LBADAPT_PATCHSIZE;
            r_offset_x = r_offset_y = r_offset_z = 1;
            w_offset_x = w_offset_y = w_offset_z = 0;
            if (4 == (dir_p4est & 4)) {
              iter_max_z = 1;
              r_offset_z = (dir_p4est % 2 == 0 ? LBADAPT_PATCHSIZE : 1);
              w_offset_z = (dir_p4est % 2 == 0 ? 0 : LBADAPT_PATCHSIZE + 1);
            } else if (2 == (dir_p4est & 2)) {
              iter_max_y = 1;
              r_offset_y = (dir_p4est % 2 == 0 ? LBADAPT_PATCHSIZE : 1);
              w_offset_y = (dir_p4est % 2 == 0 ? 0 : LBADAPT_PATCHSIZE + 1);
            } else {
              iter_max_x = 1;
              r_offset_x = (dir_p4est % 2 == 0 ? LBADAPT_PATCHSIZE : 1);
              w_offset_x = (dir_p4est % 2 == 0 ? 0 : LBADAPT_PATCHSIZE + 1);
            }

          } else if (P8EST_FACES <= dir_p4est &&
                     dir_p4est < (P8EST_FACES + P8EST_EDGES)) {
            // for edges:
            // The edge is parallel to the direction it is associated with. That
            // means for populating the halo of the patch we have to iterate
            // over this very direction while keeping both other directions
            // constant.
            iter_max_x = iter_max_y = iter_max_z = 1;
            r_offset_x = r_offset_y = r_offset_z = 1;
            w_offset_x = w_offset_y = w_offset_z = 0;
            int tmp_dir = dir_p4est - P8EST_FACES;
            int main_dir = tmp_dir / 4;
            int fc = tmp_dir % 4;
            if (0 == main_dir) {
              iter_max_x = LBADAPT_PATCHSIZE;
              r_offset_x = w_offset_x = 1;
              switch (fc) {
              case 0:
                r_offset_y = r_offset_z = LBADAPT_PATCHSIZE;
                break;
              case 1:
                r_offset_z = LBADAPT_PATCHSIZE;
                w_offset_y = LBADAPT_PATCHSIZE + 1;
                break;
              case 2:
                r_offset_y = LBADAPT_PATCHSIZE;
                w_offset_z = LBADAPT_PATCHSIZE + 1;
                break;
              case 3:
                w_offset_y = w_offset_z = LBADAPT_PATCHSIZE + 1;
                break;
              default:
                SC_ABORT_NOT_REACHED();
              }
            } else if (1 == main_dir) {
              iter_max_y = LBADAPT_PATCHSIZE;
              r_offset_y = w_offset_y = 1;
              switch (fc) {
              case 0:
                r_offset_x = r_offset_z = LBADAPT_PATCHSIZE;
                break;
              case 1:
                r_offset_z = LBADAPT_PATCHSIZE;
                w_offset_x = LBADAPT_PATCHSIZE + 1;
                break;
              case 2:
                r_offset_x = LBADAPT_PATCHSIZE;
                w_offset_z = LBADAPT_PATCHSIZE + 1;
              case 3:
                w_offset_x = w_offset_z = LBADAPT_PATCHSIZE + 1;
                break;
              default:
                SC_ABORT_NOT_REACHED();
              }
            } else if (2 == main_dir) {
              iter_max_z = LBADAPT_PATCHSIZE;
              r_offset_z = w_offset_z = 1;
              switch (fc) {
              case 0:
                r_offset_x = r_offset_y = LBADAPT_PATCHSIZE;
                break;
              case 1:
                r_offset_y = LBADAPT_PATCHSIZE;
                w_offset_x = LBADAPT_PATCHSIZE + 1;
                break;
              case 2:
                r_offset_x = LBADAPT_PATCHSIZE;
                w_offset_y = LBADAPT_PATCHSIZE + 1;
              case 3:
                w_offset_x = w_offset_y = LBADAPT_PATCHSIZE + 1;
                break;
              default:
                SC_ABORT_NOT_REACHED();
              }
            } else {
              SC_ABORT_NOT_REACHED();
            }
          } else {
            SC_ABORT_NOT_REACHED();
          }

          // for dealing with arbitrary orientations and arbitrary neighbor
          // relations: copy first to intermediate array and fill halo in
          // current patch from that temporary storage.
          // TODO: Implement that, at least in theory.

          // perform the actual data replication
          for (int patch_z = 0; patch_z < iter_max_z; ++patch_z) {
            for (int patch_y = 0; patch_y < iter_max_y; ++patch_y) {
              for (int patch_x = 0; patch_x < iter_max_x; ++patch_x) {
                memcpy(&data->patch[w_offset_x + patch_x][w_offset_y + patch_y]
                                   [w_offset_z + patch_z],
                       &neighbor_data
                            ->patch[r_offset_x + patch_x][r_offset_y + patch_y]
                                   [r_offset_z + patch_z],
                       sizeof(lbadapt_patch_cell_t));
              }
            }
          }
        }
      }
    }
  }
}
#endif // LB_ADAPTIVE_GPU

/*** Load Balance ***/
int lbadapt_partition_weight_uniform(p8est_t *p8est, p4est_topidx_t which_tree,
                                     p8est_quadrant_t *q) {
  return 1;
}

int lbadapt_partition_weight_subcycling(p8est_t *p8est,
                                        p4est_topidx_t which_tree,
                                        p8est_quadrant_t *q) {
  return (1 << (q->level - p4est_params.min_ref_level));
}

/*** REFINEMENT ***/
int refine_uniform(p8est_t *p8est, p4est_topidx_t which_tree,
                   p8est_quadrant_t *quadrant) {
  return 1;
}

int refine_random(p8est_t *p8est, p4est_topidx_t which_tree,
                  p8est_quadrant_t *quadrant) {
  return rand() % 2;
}

int refine_regional(p8est_t *p8est, p4est_topidx_t which_tree,
                    p8est_quadrant_t *q) {
  std::array<double, 3> midpoint{};
  p4est_utils_get_midpoint(p8est, which_tree, q, midpoint.data());
  if ((coords_for_regional_refinement[0] <= midpoint[0]) &&
      (midpoint[0] <= coords_for_regional_refinement[1]) &&
      (coords_for_regional_refinement[2] <= midpoint[1]) &&
      (midpoint[1] <= coords_for_regional_refinement[3]) &&
      (coords_for_regional_refinement[4] <= midpoint[2]) &&
      (midpoint[2] <= coords_for_regional_refinement[5])) {
    return 1;
  }
  return 0;
}

int coarsen_regional(p8est_t *p8est, p4est_topidx_t which_tree,
                     p8est_quadrant_t **quads) {
  std::array<double, 3> midpoint{};
  int coarsen = 1;
  for (int i = 0; i < P8EST_CHILDREN; ++i) {
    p4est_utils_get_midpoint(p8est, which_tree, quads[i], midpoint.data());
    if ((coords_for_regional_refinement[0] <= midpoint[0]) &&
        (midpoint[0] <= coords_for_regional_refinement[1]) &&
        (coords_for_regional_refinement[2] <= midpoint[1]) &&
        (midpoint[1] <= coords_for_regional_refinement[3]) &&
        (coords_for_regional_refinement[4] <= midpoint[2]) &&
        (midpoint[2] <= coords_for_regional_refinement[5])) {
      coarsen &= 1;
    } else {
      coarsen = 0;
    }
  }
  return coarsen;
}

int refine_geometric(p8est_t *p8est, p4est_topidx_t which_tree,
                     p8est_quadrant_t *q) {
  // 0.6 instead of 0.5 for stability reasons
  lb_float half_length = 0.6 * sqrt(3) * p4est_params.h[q->level];

  std::array<double, 3> midpoint{}, dist_vec{};
  p4est_utils_get_midpoint(p8est, which_tree, q, midpoint.data());

  double dist, dist_tmp;
  dist = std::numeric_limits<double>::max();

  int n = 0;
  for (auto it = LBBoundaries::lbboundaries.begin();
       it != LBBoundaries::lbboundaries.end(); ++it, ++n) {

    if (!LBBoundaries::exclude_in_geom_ref.empty()) {
      auto search_it = std::find(LBBoundaries::exclude_in_geom_ref.begin(),
                                 LBBoundaries::exclude_in_geom_ref.end(), n);
      if (search_it != LBBoundaries::exclude_in_geom_ref.end()) {
        continue;
      }
    }
    (**it).calc_dist(midpoint.data(), &dist_tmp, dist_vec.data());

    if (dist_tmp < dist) {
      dist = dist_tmp;
    }
  }

  if ((std::abs(dist) <= half_length) && !LBBoundaries::lbboundaries.empty()) {
    return 1;
  } else {
    return 0;
  }
}

int refine_inv_geometric(p8est_t *p8est, p4est_topidx_t which_tree,
                         p8est_quadrant_t *q) {
  return !refine_geometric(p8est, which_tree, q);
}

/*** HELPER FUNCTIONS ***/
int lbadapt_calc_n_from_rho_j_pi(lb_float datafield[2][19], lb_float rho,
                                 lb_float *j, const std::array<lb_float, 6> &pi,
                                 lb_float local_h) {
  int i;
  lb_float local_rho, local_j[3], local_pi[6], trace;
  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  const lb_float avg_rho = lbpar.rho * h_max * h_max * h_max;

  local_rho = rho;

  for (i = 0; i < 3; ++i) {
    local_j[i] = j[i];
  }

  for (i = 0; i < 6; ++i) {
    local_pi[i] = pi[i];
  }

  trace = local_pi[0] + local_pi[2] + local_pi[5];

  lb_float rho_times_coeff;
  lb_float tmp1, tmp2;

  /* update the q=0 sublattice */
  datafield[0][0] = 1. / 3. * (local_rho - avg_rho) - 0.5 * trace;

  /* update the q=1 sublattice */
  rho_times_coeff = 1. / 18. * (local_rho - avg_rho);

  datafield[0][1] = rho_times_coeff + 1. / 6. * local_j[0] +
                    0.25 * local_pi[0] - 1. / 12. * trace;
  datafield[0][2] = rho_times_coeff - 1. / 6. * local_j[0] +
                    0.25 * local_pi[0] - 1. / 12. * trace;
  datafield[0][3] = rho_times_coeff + 1. / 6. * local_j[1] +
                    0.25 * local_pi[2] - 1. / 12. * trace;
  datafield[0][4] = rho_times_coeff - 1. / 6. * local_j[1] +
                    0.25 * local_pi[2] - 1. / 12. * trace;
  datafield[0][5] = rho_times_coeff + 1. / 6. * local_j[2] +
                    0.25 * local_pi[5] - 1. / 12. * trace;
  datafield[0][6] = rho_times_coeff - 1. / 6. * local_j[2] +
                    0.25 * local_pi[5] - 1. / 12. * trace;

  /* update the q=2 sublattice */
  rho_times_coeff = 1. / 36. * (local_rho - avg_rho);

  tmp1 = local_pi[0] + local_pi[2];
  tmp2 = 2.0 * local_pi[1];

  datafield[0][7] = rho_times_coeff + 1. / 12. * (local_j[0] + local_j[1]) +
                    0.125 * (tmp1 + tmp2) - 1. / 24. * trace;
  datafield[0][8] = rho_times_coeff - 1. / 12. * (local_j[0] + local_j[1]) +
                    0.125 * (tmp1 + tmp2) - 1. / 24. * trace;
  datafield[0][9] = rho_times_coeff + 1. / 12. * (local_j[0] - local_j[1]) +
                    0.125 * (tmp1 - tmp2) - 1. / 24. * trace;
  datafield[0][10] = rho_times_coeff - 1. / 12. * (local_j[0] - local_j[1]) +
                     0.125 * (tmp1 - tmp2) - 1. / 24. * trace;

  tmp1 = local_pi[0] + local_pi[5];
  tmp2 = 2.0 * local_pi[3];

  datafield[0][11] = rho_times_coeff + 1. / 12. * (local_j[0] + local_j[2]) +
                     0.125 * (tmp1 + tmp2) - 1. / 24. * trace;
  datafield[0][12] = rho_times_coeff - 1. / 12. * (local_j[0] + local_j[2]) +
                     0.125 * (tmp1 + tmp2) - 1. / 24. * trace;
  datafield[0][13] = rho_times_coeff + 1. / 12. * (local_j[0] - local_j[2]) +
                     0.125 * (tmp1 - tmp2) - 1. / 24. * trace;
  datafield[0][14] = rho_times_coeff - 1. / 12. * (local_j[0] - local_j[2]) +
                     0.125 * (tmp1 - tmp2) - 1. / 24. * trace;

  tmp1 = local_pi[2] + local_pi[5];
  tmp2 = 2.0 * local_pi[4];

  datafield[0][15] = rho_times_coeff + 1. / 12. * (local_j[1] + local_j[2]) +
                     0.125 * (tmp1 + tmp2) - 1. / 24. * trace;
  datafield[0][16] = rho_times_coeff - 1. / 12. * (local_j[1] + local_j[2]) +
                     0.125 * (tmp1 + tmp2) - 1. / 24. * trace;
  datafield[0][17] = rho_times_coeff + 1. / 12. * (local_j[1] - local_j[2]) +
                     0.125 * (tmp1 - tmp2) - 1. / 24. * trace;
  datafield[0][18] = rho_times_coeff - 1. / 12. * (local_j[1] - local_j[2]) +
                     0.125 * (tmp1 - tmp2) - 1. / 24. * trace;

  return 0;
}

int lbadapt_calc_local_fields(lb_float populations[2][19], lb_float force[3],
                              int boundary, int has_force, lb_float local_h,
                              lb_float *rho, lb_float *j, lb_float *pi) {
  int level = log2((lb_float)(P8EST_ROOT_LEN >> P8EST_MAXLEVEL) / local_h);
  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];
#ifdef LB_BOUNDARIES
  if (boundary) {
    // set all to 0 on boundary
    *rho = lbpar.rho * h_max * h_max * h_max;
    j[0] = 0.;
    j[1] = 0.;
    j[2] = 0.;
    if (pi) {
      pi[0] = 0.;
      pi[1] = 0.;
      pi[2] = 0.;
      pi[3] = 0.;
      pi[4] = 0.;
      pi[5] = 0.;
    }
    return 0;
  }
#endif // LB_BOUNDARIES

  lb_float cpmode[19];
  lbadapt_calc_modes(populations, cpmode);

  lb_float modes_from_pi_eq[6];

  *rho = cpmode[0] + lbpar.rho * h_max * h_max * h_max;

  j[0] = cpmode[1];
  j[1] = cpmode[2];
  j[2] = cpmode[3];

#ifndef EXTERNAL_FORCES
  if (has_force_density)
#endif // EXTERNAL_FORCES
  {
    j[0] += 0.5 * force[0];
    j[1] += 0.5 * force[1];
    j[2] += 0.5 * force[2];
  }
  if (!pi)
    return 0;

  /* equilibrium part of the stress modes */
  modes_from_pi_eq[0] = scalar(j, j) / *rho;
  modes_from_pi_eq[1] = (Utils::sqr(j[0]) - Utils::sqr(j[1])) / *rho;
  modes_from_pi_eq[2] = (scalar(j, j) - 3.0 * Utils::sqr(j[2])) / *rho;
  modes_from_pi_eq[3] = j[0] * j[1] / *rho;
  modes_from_pi_eq[4] = j[0] * j[2] / *rho;
  modes_from_pi_eq[5] = j[1] * j[2] / *rho;

  /* Now we must predict the outcome of the next collision */
  /* We immediately average pre- and post-collision. */
  cpmode[4] = modes_from_pi_eq[0] + (0.5 + 0.5 * lbpar.gamma_bulk[level]) *
                                        (cpmode[4] - modes_from_pi_eq[0]);
  cpmode[5] = modes_from_pi_eq[1] + (0.5 + 0.5 * lbpar.gamma_shear[level]) *
                                        (cpmode[5] - modes_from_pi_eq[1]);
  cpmode[6] = modes_from_pi_eq[2] + (0.5 + 0.5 * lbpar.gamma_shear[level]) *
                                        (cpmode[6] - modes_from_pi_eq[2]);
  cpmode[7] = modes_from_pi_eq[3] + (0.5 + 0.5 * lbpar.gamma_shear[level]) *
                                        (cpmode[7] - modes_from_pi_eq[3]);
  cpmode[8] = modes_from_pi_eq[4] + (0.5 + 0.5 * lbpar.gamma_shear[level]) *
                                        (cpmode[8] - modes_from_pi_eq[4]);
  cpmode[9] = modes_from_pi_eq[5] + (0.5 + 0.5 * lbpar.gamma_shear[level]) *
                                        (cpmode[9] - modes_from_pi_eq[5]);

  // Transform the stress tensor components according to the modes that
  // correspond to those used by U. Schiller. In terms of populations this
  // expression then corresponds exactly to those in Eqs. 116 - 121 in the
  // Duenweg and Ladd paper, when these are written out in populations.
  // But to ensure this, the expression in Schiller's modes has to be
  // different!

  pi[0] =
      (2.0 * (cpmode[0] + cpmode[4]) + cpmode[6] + 3.0 * cpmode[5]) / 6.0; // xx
  pi[1] = cpmode[7];                                                       // xy
  pi[2] =
      (2.0 * (cpmode[0] + cpmode[4]) + cpmode[6] - 3.0 * cpmode[5]) / 6.0; // yy
  pi[3] = cpmode[8];                                                       // xz
  pi[4] = cpmode[9];                                                       // yz
  pi[5] = (cpmode[0] + cpmode[4] - cpmode[6]) / 3.0;                       // zz

  return 0;
}

int lbadapt_calc_modes(lb_float population[2][19], lb_float *mode) {
#ifndef LB_ADAPTIVE_GPU
  lb_float n0, n1p, n1m, n2p, n2m, n3p, n3m, n4p, n4m, n5p, n5m, n6p, n6m, n7p,
      n7m, n8p, n8m, n9p, n9m;

  // clang-format off
  n0  = population[0][ 0];
  n1p = population[0][ 1] + population[0][ 2];
  n1m = population[0][ 1] - population[0][ 2];
  n2p = population[0][ 3] + population[0][ 4];
  n2m = population[0][ 3] - population[0][ 4];
  n3p = population[0][ 5] + population[0][ 6];
  n3m = population[0][ 5] - population[0][ 6];
  n4p = population[0][ 7] + population[0][ 8];
  n4m = population[0][ 7] - population[0][ 8];
  n5p = population[0][ 9] + population[0][10];
  n5m = population[0][ 9] - population[0][10];
  n6p = population[0][11] + population[0][12];
  n6m = population[0][11] - population[0][12];
  n7p = population[0][13] + population[0][14];
  n7m = population[0][13] - population[0][14];
  n8p = population[0][15] + population[0][16];
  n8m = population[0][15] - population[0][16];
  n9p = population[0][17] + population[0][18];
  n9m = population[0][17] - population[0][18];
  // clang-format on

  /* mass mode */
  mode[0] = n0 + n1p + n2p + n3p + n4p + n5p + n6p + n7p + n8p + n9p;

  /* momentum modes */
  mode[1] = n1m + n4m + n5m + n6m + n7m;
  mode[2] = n2m + n4m - n5m + n8m + n9m;
  mode[3] = n3m + n6m - n7m + n8m - n9m;

  /* stress modes */
  mode[4] = -n0 + n4p + n5p + n6p + n7p + n8p + n9p;
  mode[5] = n1p - n2p + n6p + n7p - n8p - n9p;
  mode[6] = n1p + n2p - n6p - n7p - n8p - n9p - 2. * (n3p - n4p - n5p);
  mode[7] = n4p - n5p;
  mode[8] = n6p - n7p;
  mode[9] = n8p - n9p;

  /* kinetic modes */
  mode[10] = -2. * n1m + n4m + n5m + n6m + n7m;
  mode[11] = -2. * n2m + n4m - n5m + n8m + n9m;
  mode[12] = -2. * n3m + n6m - n7m + n8m - n9m;
  mode[13] = n4m + n5m - n6m - n7m;
  mode[14] = n4m - n5m - n8m - n9m;
  mode[15] = n6m - n7m - n8m + n9m;
  mode[16] = n0 + n4p + n5p + n6p + n7p + n8p + n9p - 2. * (n1p + n2p + n3p);
  mode[17] = -n1p + n2p + n6p + n7p - n8p - n9p;
  mode[18] = -n1p - n2p - n6p - n7p - n8p - n9p + 2. * (n3p + n4p + n5p);
#endif // LB_ADAPTIVE_GPU

  return 0;
}

int lbadapt_relax_modes(lb_float *mode, lb_float *force, lb_float local_h) {
#ifndef LB_ADAPTIVE_GPU
  lb_float rho, j[3], pi_eq[6];

  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  int level = log2((lb_float)(P8EST_ROOT_LEN >> P8EST_MAXLEVEL) / local_h);

  /* re-construct the real density
   * remember that the populations are stored as differences to their
   * equilibrium value */
  rho = mode[0] + lbpar.rho * h_max * h_max * h_max;

  j[0] = mode[1] + 0.5 * force[0];
  j[1] = mode[2] + 0.5 * force[1];
  j[2] = mode[3] + 0.5 * force[2];

  /* equilibrium part of the stress modes */
  pi_eq[0] = scalar(j, j) / rho;
  pi_eq[1] = (Utils::sqr(j[0]) - Utils::sqr(j[1])) / rho;
  pi_eq[2] = (scalar(j, j) - 3.0 * Utils::sqr(j[2])) / rho;
  pi_eq[3] = j[0] * j[1] / rho;
  pi_eq[4] = j[0] * j[2] / rho;
  pi_eq[5] = j[1] * j[2] / rho;

  /* relax the stress modes */
  // clang-format off
  mode[4] = pi_eq[0] + lbpar.gamma_bulk[level]  * (mode[4] - pi_eq[0]);
  mode[5] = pi_eq[1] + lbpar.gamma_shear[level] * (mode[5] - pi_eq[1]);
  mode[6] = pi_eq[2] + lbpar.gamma_shear[level] * (mode[6] - pi_eq[2]);
  mode[7] = pi_eq[3] + lbpar.gamma_shear[level] * (mode[7] - pi_eq[3]);
  mode[8] = pi_eq[4] + lbpar.gamma_shear[level] * (mode[8] - pi_eq[4]);
  mode[9] = pi_eq[5] + lbpar.gamma_shear[level] * (mode[9] - pi_eq[5]);
  // clang-format on

  /* relax the ghost modes (project them out) */
  /* ghost modes have no equilibrium part due to orthogonality */
  // clang-format off
  mode[10] = lbpar.gamma_odd  * mode[10];
  mode[11] = lbpar.gamma_odd  * mode[11];
  mode[12] = lbpar.gamma_odd  * mode[12];
  mode[13] = lbpar.gamma_odd  * mode[13];
  mode[14] = lbpar.gamma_odd  * mode[14];
  mode[15] = lbpar.gamma_odd  * mode[15];
  mode[16] = lbpar.gamma_even * mode[16];
  mode[17] = lbpar.gamma_even * mode[17];
  mode[18] = lbpar.gamma_even * mode[18];
  // clang-format on

#endif // LB_ADAPTIVE_GPU

  return 0;
}

int lbadapt_thermalize_modes(lb_float *mode) {
  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];
  const double rootrho =
      std::sqrt(std::fabs(mode[0] + lbpar.rho * h_max * h_max * h_max));

#ifdef GAUSSRANDOM
  constexpr double variance = 1. / 12.;
  auto rng = []() -> double { return gaussian_random(); };
#elif defined(GAUSSRANDOMCUT)
  constexpr double variance = 1.0;
  auto rng = []() -> double { return gaussian_random_cut(); };
#elif defined(FLATNOISE)
  constexpr double variance = 1. / 12.0;
  auto rng = []() -> double { return d_random() - 0.5; };
#else // GAUSSRANDOM
#error No noise type defined for the CPU LB
#endif // GAUSSRANDOM

  auto const pref = std::sqrt(1. / variance) * rootrho;

  /* stress modes */
  mode[4] += pref * lbpar.phi[4] * rng();
  mode[5] += pref * lbpar.phi[5] * rng();
  mode[6] += pref * lbpar.phi[6] * rng();
  mode[7] += pref * lbpar.phi[7] * rng();
  mode[8] += pref * lbpar.phi[8] * rng();
  mode[9] += pref * lbpar.phi[9] * rng();

  /* ghost modes */
  mode[10] += pref * lbpar.phi[10] * rng();
  mode[11] += pref * lbpar.phi[11] * rng();
  mode[12] += pref * lbpar.phi[12] * rng();
  mode[13] += pref * lbpar.phi[13] * rng();
  mode[14] += pref * lbpar.phi[14] * rng();
  mode[15] += pref * lbpar.phi[15] * rng();
  mode[16] += pref * lbpar.phi[16] * rng();
  mode[17] += pref * lbpar.phi[17] * rng();
  mode[18] += pref * lbpar.phi[18] * rng();

#ifdef ADDITIONAL_CHECKS
  rancounter += 15;
#endif // ADDITIONAL_CHECKS

  return 0;
}

int lbadapt_apply_forces(lb_float *mode, lb_float *f, lb_float local_h) {
  lb_float rho, u[3], C[6];

  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  int level = log2((lb_float)(P8EST_ROOT_LEN >> P8EST_MAXLEVEL) / local_h);

  rho = mode[0] + lbpar.rho * h_max * h_max * h_max;

  /* hydrodynamic momentum density is redefined when external forces present
   */
  u[0] = (mode[1] + 0.5 * f[0]) / rho;
  u[1] = (mode[2] + 0.5 * f[1]) / rho;
  u[2] = (mode[3] + 0.5 * f[2]) / rho;

  C[0] = (1. + lbpar.gamma_bulk[level]) * u[0] * f[0] +
         1. / 3. * (lbpar.gamma_bulk[level] - lbpar.gamma_shear[level]) *
             scalar(u, f);
  C[2] = (1. + lbpar.gamma_bulk[level]) * u[1] * f[1] +
         1. / 3. * (lbpar.gamma_bulk[level] - lbpar.gamma_shear[level]) *
             scalar(u, f);
  C[5] = (1. + lbpar.gamma_bulk[level]) * u[2] * f[2] +
         1. / 3. * (lbpar.gamma_bulk[level] - lbpar.gamma_shear[level]) *
             scalar(u, f);
  C[1] = 0.5 * (1. + lbpar.gamma_shear[level]) * (u[0] * f[1] + u[1] * f[0]);
  C[3] = 0.5 * (1. + lbpar.gamma_shear[level]) * (u[0] * f[2] + u[2] * f[0]);
  C[4] = 0.5 * (1. + lbpar.gamma_shear[level]) * (u[1] * f[2] + u[2] * f[1]);

  /* update momentum modes */
  mode[1] += f[0];
  mode[2] += f[1];
  mode[3] += f[2];

  /* update stress modes */
  mode[4] += C[0] + C[2] + C[5];
  mode[5] += C[0] - C[2];
  mode[6] += C[0] + C[2] - 2. * C[5];
  mode[7] += C[1];
  mode[8] += C[3];
  mode[9] += C[4];

// reset force to external force (remove influences from particle coupling)
#ifdef EXTERNAL_FORCES
  // unit conversion: force density
  f[0] = p4est_params.prefactors[level] * lbpar.ext_force_density[0] *
         Utils::sqr(h_max) * Utils::sqr(lbpar.tau);
  f[1] = p4est_params.prefactors[level] * lbpar.ext_force_density[1] *
         Utils::sqr(h_max) * Utils::sqr(lbpar.tau);
  f[2] = p4est_params.prefactors[level] * lbpar.ext_force_density[2] *
         Utils::sqr(h_max) * Utils::sqr(lbpar.tau);
#else  // EXTERNAL_FORCES
  f[0] = 0.0;
  f[1] = 0.0;
  f[2] = 0.0;
#endif // EXTERNAL_FORCES

  return 0;
}

lb_float lbadapt_backTransformation(lb_float *m, int dir) {
  switch (dir) {
  case 0:
    return m[0] - m[4] + m[16];
  case 1:
    return m[0] + m[1] + m[5] + m[6] - m[17] - m[18] - 2. * (m[10] + m[16]);
  case 2:
    return m[0] - m[1] + m[5] + m[6] - m[17] - m[18] + 2. * (m[10] - m[16]);
  case 3:
    return m[0] + m[2] - m[5] + m[6] + m[17] - m[18] - 2. * (m[11] + m[16]);
  case 4:
    return m[0] - m[2] - m[5] + m[6] + m[17] - m[18] + 2. * (m[11] - m[16]);
  case 5:
    return m[0] + m[3] - 2. * (m[6] + m[12] + m[16] - m[18]);
  case 6:
    return m[0] - m[3] - 2. * (m[6] - m[12] + m[16] - m[18]);
  case 7:
    return m[0] + m[1] + m[2] + m[4] + 2. * m[6] + m[7] + m[10] + m[11] +
           m[13] + m[14] + m[16] + 2. * m[18];
  case 8:
    return m[0] - m[1] - m[2] + m[4] + 2. * m[6] + m[7] - m[10] - m[11] -
           m[13] - m[14] + m[16] + 2. * m[18];
  case 9:
    return m[0] + m[1] - m[2] + m[4] + 2. * m[6] - m[7] + m[10] - m[11] +
           m[13] - m[14] + m[16] + 2. * m[18];
  case 10:
    return m[0] - m[1] + m[2] + m[4] + 2. * m[6] - m[7] - m[10] + m[11] -
           m[13] + m[14] + m[16] + 2. * m[18];
  case 11:
    return m[0] + m[1] + m[3] + m[4] + m[5] - m[6] + m[8] + m[10] + m[12] -
           m[13] + m[15] + m[16] + m[17] - m[18];
  case 12:
    return m[0] - m[1] - m[3] + m[4] + m[5] - m[6] + m[8] - m[10] - m[12] +
           m[13] - m[15] + m[16] + m[17] - m[18];
  case 13:
    return m[0] + m[1] - m[3] + m[4] + m[5] - m[6] - m[8] + m[10] - m[12] -
           m[13] - m[15] + m[16] + m[17] - m[18];
  case 14:
    return m[0] - m[1] + m[3] + m[4] + m[5] - m[6] - m[8] - m[10] + m[12] +
           m[13] + m[15] + m[16] + m[17] - m[18];
  case 15:
    return m[0] + m[2] + m[3] + m[4] - m[5] - m[6] + m[9] + m[11] + m[12] -
           m[14] - m[15] + m[16] - m[17] - m[18];
  case 16:
    return m[0] - m[2] - m[3] + m[4] - m[5] - m[6] + m[9] - m[11] - m[12] +
           m[14] + m[15] + m[16] - m[17] - m[18];
  case 17:
    return m[0] + m[2] - m[3] + m[4] - m[5] - m[6] - m[9] + m[11] - m[12] -
           m[14] + m[15] + m[16] - m[17] - m[18];
  case 18:
    return m[0] - m[2] + m[3] + m[4] - m[5] - m[6] - m[9] - m[11] + m[12] +
           m[14] - m[15] + m[16] - m[17] - m[18];
  default:
    SC_ABORT_NOT_REACHED();
  }
}

void lbadapt_pass_populations(p8est_meshiter_t *mesh_iter,
                              lbadapt_payload_t *currCellData) {
#ifndef LB_ADAPTIVE_GPU
  // clang-format off
  int inv[] = { 0,
                2,  1,  4,  3,  6,  5,
                8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17 };
  // clang-format on
  lbadapt_payload_t *data;

  // copy resting population
  currCellData->lbfluid[1][0] = currCellData->lbfluid[0][0];

  // stream to surrounding cells
  for (int dir_ESPR = 1; dir_ESPR < LB_Model<>::n_veloc; ++dir_ESPR) {
    // set neighboring cell information in iterator
    int dir_p4est = ci_to_p4est[(dir_ESPR - 1)];
    p8est_meshiter_set_neighbor_quad_info(mesh_iter, dir_p4est);

    int inv_neigh_dir_p4est = mesh_iter->neighbor_entity_index;
    // stream if we found a neighbor.
    // however, do not stream between two real quadrants that both hold virtual
    // children
#if 0
    if (mesh_iter->neighbor_qid != -1
        && ((-1 != mesh_iter->current_vid)
            || (-1 != mesh_iter->neighbor_vid)
            || (-1 == adapt_virtual->virtual_qflags[mesh_iter->current_qid])
            || (!mesh_iter->neighbor_is_ghost
                && -1 ==
                   adapt_virtual->virtual_qflags[mesh_iter->neighbor_qid])
            || (mesh_iter->neighbor_is_ghost
                && -1 ==
                   adapt_virtual->virtual_gflags[mesh_iter->neighbor_qid])))
#else  // 0
    if (mesh_iter->neighbor_qid != -1)
#endif // 0
    {
      int inv_neigh_dir_ESPR = p4est_to_ci[inv_neigh_dir_p4est];
      assert(inv[dir_ESPR] == inv_neigh_dir_ESPR);
      assert(dir_ESPR == inv[inv_neigh_dir_ESPR]);

      if (mesh_iter->neighbor_is_ghost) {
        data = &lbadapt_ghost_data[mesh_iter->current_level].at(
            p8est_meshiter_get_neighbor_storage_id(mesh_iter));
        if (!data->lbfields.boundary) {
          // invert streaming step only if ghost neighbor is no boundary cell
          currCellData->lbfluid[1][inv[dir_ESPR]] =
              data->lbfluid[0][inv_neigh_dir_ESPR];
        }
      } else {
        data = &lbadapt_local_data[mesh_iter->current_level].at(
            p8est_meshiter_get_neighbor_storage_id(mesh_iter));
        data->lbfluid[1][inv[inv_neigh_dir_ESPR]] =
            currCellData->lbfluid[0][dir_ESPR];
      }
    }
  }
#endif // LB_ADAPTIVE_GPU
}

int lbadapt_calc_pop_from_modes(lb_float *populations, lb_float *m) {
#ifdef LB_ADAPTIVE
  /* normalization factors enter in the back transformation */
  for (int i = 0; i < LB_Model<>::n_veloc; ++i) {
    m[i] *= (1. / d3q19_modebase[19][i]);
  }

  /** perform back transformation and add weights */
  for (int i = 0; i < LB_Model<>::n_veloc; ++i) {
    populations[i] = lbadapt_backTransformation(m, i) * lbmodel.w[i];
  }
#endif // LB_ADAPTIVE
  return 0;
}

void lbadapt_collide(int level, p8est_meshiter_localghost_t quads_to_collide) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
#ifdef LB_ADAPTIVE_GPU
  lb_float h = (lb_float)P8EST_QUADRANT_LEN(level) /
               ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h = (lb_float)P8EST_QUADRANT_LEN(level) / (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  lbadapt_payload_t *data;
  bool has_virtuals;

  lb_float modes[19];
  castable_unique_ptr<p4est_meshiter_t> mesh_iter = p8est_meshiter_new_ext(
      adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
      P8EST_CONNECT_EDGE, quads_to_collide, P8EST_TRAVERSE_REAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      if (mesh_iter->current_is_ghost) {
        data = &lbadapt_ghost_data[level].at(
            p8est_meshiter_get_current_storage_id(mesh_iter));
        has_virtuals =
            (-1 !=
             mesh_iter->virtual_quads->virtual_gflags[mesh_iter->current_qid]);
      } else {
        data = &lbadapt_local_data[level].at(
            p8est_meshiter_get_current_storage_id(mesh_iter));
        has_virtuals =
            (-1 !=
             mesh_iter->virtual_quads->virtual_qflags[mesh_iter->current_qid]);
      }
#ifdef LB_BOUNDARIES
      if (!data->lbfields.boundary)
#endif // LB_BOUNDARIES
      {
        /* calculate modes locally */
        lbadapt_calc_modes(data->lbfluid, modes);

        /* deterministic collisions */
        lbadapt_relax_modes(modes, data->lbfields.force_density, h);

        /* fluctuating hydrodynamics */
        if (lbpar.fluct) {
          lbadapt_thermalize_modes(modes);
        }

        // apply forces
        lbadapt_apply_forces(modes, data->lbfields.force_density, h);

        // perform back transformation
        lbadapt_calc_pop_from_modes(data->lbfluid[0], modes);
      }
      if (has_virtuals) {
        lbadapt_populate_virtuals(mesh_iter);
      }
    }
  }
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_populate_virtuals(p8est_meshiter_t *mesh_iter) {
#ifndef LB_ADAPTIVE_GPU
  int parent_qid = mesh_iter->current_qid;
  int virtual_sid;
  int lvl = 1 + mesh_iter->current_level;
  lbadapt_payload_t *source_data;
  lbadapt_payload_t *virtual_data;

  // virtual quads are local if their parent is local, ghost analogous
  bool is_ghost = mesh_iter->current_is_ghost;

  if (is_ghost) {
    virtual_sid = mesh_iter->virtual_quads->quad_gvirtual_offset[parent_qid];
    source_data = &lbadapt_ghost_data[mesh_iter->current_level].at(
        p8est_meshiter_get_current_storage_id(mesh_iter));
    virtual_data = &lbadapt_ghost_data[lvl].at(virtual_sid);
  } else {
    virtual_sid = mesh_iter->virtual_quads->quad_qvirtual_offset[parent_qid];
    source_data = &lbadapt_local_data[mesh_iter->current_level].at(
        p8est_meshiter_get_current_storage_id(mesh_iter));
    virtual_data = &lbadapt_local_data[lvl].at(virtual_sid);
  }
  for (int i = 0; i < P8EST_CHILDREN; ++i) {
    std::memcpy(virtual_data->lbfluid[0], source_data->lbfluid[0],
                LB_Model<>::n_veloc * sizeof(lb_float));
    std::memcpy(virtual_data->lbfluid[1], virtual_data->lbfluid[0],
                LB_Model<>::n_veloc * sizeof(lb_float));
    // make sure that boundary is not occupied by some memory clutter
    virtual_data->lbfields.boundary = source_data->lbfields.boundary;
    ++virtual_data;
  }
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_stream(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  lbadapt_payload_t *data;
  castable_unique_ptr<p4est_meshiter_t> mesh_iter = p4est_meshiter_new_ext(
      adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
      P8EST_CONNECT_EDGE, P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REALVIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      data = &lbadapt_local_data[level].at(
          p8est_meshiter_get_current_storage_id(mesh_iter));
      if (!data->lbfields.boundary) {
        lbadapt_pass_populations(mesh_iter, data);
      }
    }
  }
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_bounce_back(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  lbadapt_payload_t *data, *currCellData;

  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  // vector of inverse c_i, 0 is inverse to itself.
  // clang-format off
  const int inv[] = { 0,
                      2,  1,  4,  3,  6,  5,
                      8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17 };
  // clang-format on

  castable_unique_ptr<p4est_meshiter_t> mesh_iter = p4est_meshiter_new_ext(
      adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
      P8EST_CONNECT_EDGE, P4EST_TRAVERSE_LOCAL, P4EST_TRAVERSE_REALVIRTUAL,
      P4EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      currCellData = &lbadapt_local_data[level].at(
          p8est_meshiter_get_current_storage_id(mesh_iter));

      lb_float population_shift;

      if (currCellData->lbfields.boundary) {
        currCellData->lbfluid[1][0] = 0.0;
      }

      /** The general idea of the implementation on a regular Cartesian grid is
       * that the grid is traversed and for each population we check if we got
       * it streamed in error.
       * E.g. f_1 (c_1: 1, 0, 0) had to be streamed from cell in the inverse
       * directon (c_2: -1, 0, 0).
       * So check if cell in inverse direction is a boundary or a fluid cell:
       * Boundary:
       * Set f_(iinv) of neighbor cell and f_i of current cell to 0.
       * Fluid:
       * Set f_(iinv) of neighbor cell to (f_i + shift) of current cell.
       *
       * We cannot copy this with minimal invasiveness, because boundary cells
       * can end in the ghost layer and p4est_iterate does not visit ghost
       * cells. Therefore we have to inspect in each cell, if it has neighbors
       * that are part of the ghost layer.
       * Then there are 2 cases where we have to bounce back:
       * 1. Current cell itself is boundary cell
       *    In this case we perform the same algorithm as in the regular grid
       * 2. Current cell is fluid cell and has ghost neighbors that are
       *    boundary cells.
       *    In this case we we have to do something different:
       *    The neighboring cell was not allowed to stream, because it is a
       *    boundary cell and in addition, we did not stream to it.
       *    So instead we have to perform the back-transformation for the
       *    populations of this cell and write our population in search
       *    direction to our inverse position.
       */
      for (int dir_ESPR = 1; dir_ESPR < 19; ++dir_ESPR) {
        // set neighboring cell information in iterator
        int inv_dir_p4est = ci_to_p4est[inv[dir_ESPR] - 1];
        p8est_meshiter_set_neighbor_quad_info(mesh_iter, inv_dir_p4est);

        // bounce back if we found a neighbor.
        // however, do not bounce back between two quadrants that hold virtual
        // children
#if 0
        if (mesh_iter->neighbor_qid != -1
            && ((-1 != mesh_iter->current_vid)
                || (-1 != mesh_iter->neighbor_vid)
                || (-1 == adapt_virtual->virtual_qflags[mesh_iter->current_qid])
                || (!mesh_iter->neighbor_is_ghost
                    && -1 ==
                       adapt_virtual->virtual_qflags[mesh_iter->neighbor_qid])
                || (mesh_iter->neighbor_is_ghost
                    && -1 ==
                       adapt_virtual->virtual_gflags[mesh_iter->neighbor_qid])))
#else  // 0
        if (mesh_iter->neighbor_qid != -1)
#endif // 0
        {
          /** read c_i in neighbors orientation and p4est's direction system.
           * Convert it to ESPResSo's directions.
           * The inverse direction of that result is the correct value to bounce
           * back to.
           */
          int neigh_dir_p4est = mesh_iter->neighbor_entity_index;
          int neigh_dir_ESPR = p4est_to_ci[neigh_dir_p4est];
          int inv_neigh_dir_ESPR = inv[neigh_dir_ESPR];

          /** in case of a brick connectivity certain symmetries hold even
           * across tree boundaries
           */
          assert(dir_ESPR == neigh_dir_ESPR);
          assert(inv[dir_ESPR] == inv_neigh_dir_ESPR);

          /** fetch data of neighbor cell */
          if (mesh_iter->neighbor_is_ghost) {
            data = &lbadapt_ghost_data[level].at(
                p8est_meshiter_get_neighbor_storage_id(mesh_iter));
          } else {
            data = &lbadapt_local_data[level].at(
                p8est_meshiter_get_neighbor_storage_id(mesh_iter));
          }

          // case 1
          if (!mesh_iter->neighbor_is_ghost &&
              currCellData->lbfields.boundary) {
            if (!data->lbfields.boundary) {
              // calculate population shift (velocity boundary condition)
              population_shift = 0.;
              for (int l = 0; l < 3; ++l) {
                population_shift -= h_max * h_max * h_max * lbpar.rho * 2 *
                                    lbmodel.c[dir_ESPR][l] *
                                    lbmodel.w[dir_ESPR] *
                                    (*LBBoundaries::lbboundaries
                                         [currCellData->lbfields.boundary - 1])
                                        .velocity()[l] *
                                    (lbpar.tau / h_max) / lbmodel.c_sound_sq;
              }

              // sum up the force that the fluid applies on the boundary
              for (int l = 0; l < 3; ++l) {
                (*LBBoundaries::lbboundaries[currCellData->lbfields.boundary -
                                             1])
                    .force()[l] += (2 * currCellData->lbfluid[1][dir_ESPR] +
                                    population_shift) *
                                   lbmodel.c[dir_ESPR][l];
              }

              // add if we bounce back from a cell without virtual quadrants
              // into a coarse cell hosting virtual quadrants
              data->lbfluid[1][inv_neigh_dir_ESPR] =
                  currCellData->lbfluid[1][dir_ESPR] + population_shift;
            } else {
              data->lbfluid[1][inv_neigh_dir_ESPR] =
                  currCellData->lbfluid[1][dir_ESPR] = 0.0;
            }
          } // if (!mesh_iter->neighbor_is_ghost && currCellData->boundary)

          // case 2
          else if (mesh_iter->neighbor_is_ghost && data->lbfields.boundary) {
            if (!currCellData->lbfields.boundary) {
              // calculate population shift
              population_shift = 0.;
              for (int l = 0; l < 3; l++) {
                population_shift -=
                    h_max * h_max * h_max * lbpar.rho * 2 *
                    lbmodel.c[inv[dir_ESPR]][l] * lbmodel.w[inv[dir_ESPR]] *
                    (*LBBoundaries::lbboundaries[data->lbfields.boundary - 1])
                        .velocity()[l] *
                    (lbpar.tau / h_max) / lbmodel.c_sound_sq;
              }

              currCellData->lbfluid[1][dir_ESPR] =
                  currCellData->lbfluid[0][inv[dir_ESPR]] + population_shift;
            } else {
              currCellData->lbfluid[1][dir_ESPR] = 0.0;
            }
          }
        } // if (mesh_iter->neighbor_qid) != -1
      }   // for (int dir_ESPR = 1; dir_ESPR < 19; ++dir_ESPR)
    }
  }
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_update_populations_from_virtuals(
    int level, p8est_meshiter_localghost_t quads_to_update) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  int parent_sid;
  lbadapt_payload_t *data, *parent_data;
  int vel;
  castable_unique_ptr<p4est_meshiter_t> mesh_iter = p8est_meshiter_new_ext(
      adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level + 1,
      P8EST_CONNECT_EDGE, quads_to_update, P8EST_TRAVERSE_VIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      P4EST_ASSERT(mesh_iter->current_index % P4EST_CHILDREN ==
                   mesh_iter->current_vid);
      // virtual quads are local iff their parent is local
      if (!mesh_iter->current_is_ghost) {
        parent_sid =
            mesh_iter->virtual_quads->quad_qreal_offset[mesh_iter->current_qid];
        data = &lbadapt_local_data.at(level + 1).at(
            p8est_meshiter_get_current_storage_id(mesh_iter));
        parent_data = &lbadapt_local_data.at(level).at(parent_sid);
      } else {
        parent_sid =
            mesh_iter->virtual_quads->quad_greal_offset[mesh_iter->current_qid];
        data = &lbadapt_ghost_data.at(level + 1).at(
            p8est_meshiter_get_current_storage_id(mesh_iter));
        parent_data = &lbadapt_ghost_data.at(level).at(parent_sid);
      }
      if (!data->lbfields.boundary) {
        if (!mesh_iter->current_vid) {
          std::fill_n(std::begin(parent_data->lbfluid[1]), LB_Model<>::n_veloc,
                      0);
        }
        for (vel = 0; vel < LB_Model<>::n_veloc; ++vel) {
          // child velocities have already been swapped here
          parent_data->lbfluid[1][vel] += 0.125 * data->lbfluid[0][vel];
        }
      }
    }
  }
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_swap_pointers(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  lbadapt_payload_t *data;
  castable_unique_ptr<p4est_meshiter_t> mesh_iter = p8est_meshiter_new_ext(
      adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
      P8EST_CONNECT_EDGE, P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REALVIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      if (!mesh_iter->current_is_ghost) {
        data = &lbadapt_local_data[level].at(
            p8est_meshiter_get_current_storage_id(mesh_iter));
      } else {
        data = &lbadapt_ghost_data[level].at(
            p8est_meshiter_get_current_storage_id(mesh_iter));
      }
      std::swap(data->lbfluid[0], data->lbfluid[1]);
    }
  }
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_get_boundary_values(sc_array_t *boundary_values) {
  const auto &forest = p4est_utils_get_forest_info(forest_order::adaptive_LB);
  int status;
  int level;
  double bnd, *bnd_ptr;
  lbadapt_payload_t *data;
#ifdef LB_ADAPTIVE_GPU
  int cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
#endif // LB_ADAPTIVE_GPU

  castable_unique_ptr<p4est_meshiter_t> mesh_iter;
  /* get boundary status */
  for (level = forest.coarsest_level_local; level <= forest.finest_level_local;
       ++level) {
    status = 0;
    mesh_iter.reset(p8est_meshiter_new_ext(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
        adapt_ghost->btype, P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER));
    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[level].at(
            p8est_meshiter_get_current_storage_id(mesh_iter));
#ifndef LB_ADAPTIVE_GPU
        /* just grab the value of each cell and pass it into solution vector */
        bnd = static_cast<double>(data->lbfields.boundary);
        bnd_ptr =
            (double *)sc_array_index(boundary_values, mesh_iter->current_qid);
        *bnd_ptr = bnd;
#else  // LB_ADAPTIVE_GPU
        bnd_ptr = (double *)sc_array_index(
            boundary_values, cells_per_patch * mesh_iter->current_qid);
        int patch_count = 0;
        for (int patch_z = 1; patch_z <= LBADAPT_PATCHSIZE; ++patch_z) {
          for (int patch_y = 1; patch_y <= LBADAPT_PATCHSIZE; ++patch_y) {
            for (int patch_x = 1; patch_x <= LBADAPT_PATCHSIZE; ++patch_x) {
              bnd_ptr[patch_count] =
                  data->patch[patch_x][patch_y][patch_z].boundary;
              ++patch_count;
            }
          }
        }
#endif // LB_ADAPTIVE_GPU
      }
    }
  }
}

void lbadapt_get_density_values(sc_array_t *density_values) {
  const auto &forest = p4est_utils_get_forest_info(forest_order::adaptive_LB);
  int status;
  int level;
  double dens, *dens_ptr;
  lbadapt_payload_t *data;

#ifdef LB_ADAPTIVE_GPU
  int cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
#endif // LB_ADAPTIVE_GPU
  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  castable_unique_ptr<p4est_meshiter_t> mesh_iter;
  for (level = forest.coarsest_level_local; level <= forest.finest_level_local;
       ++level) {
    status = 0;
    mesh_iter.reset(p8est_meshiter_new_ext(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
        adapt_ghost->btype, P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER));

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[level].at(
            p8est_meshiter_get_current_storage_id(mesh_iter));

        double avg_rho = lbpar.rho * h_max * h_max * h_max;

#ifndef LB_ADAPTIVE_GPU
        if (data->lbfields.boundary) {
          dens = 0;
        } else {
          // clang-format off
          dens = static_cast<double>(avg_rho
               + data->lbfluid[0][ 0] + data->lbfluid[0][ 1]
               + data->lbfluid[0][ 2] + data->lbfluid[0][ 3]
               + data->lbfluid[0][ 4] + data->lbfluid[0][ 5]
               + data->lbfluid[0][ 6] + data->lbfluid[0][ 7]
               + data->lbfluid[0][ 8] + data->lbfluid[0][ 9]
               + data->lbfluid[0][10] + data->lbfluid[0][11]
               + data->lbfluid[0][12] + data->lbfluid[0][13]
               + data->lbfluid[0][14] + data->lbfluid[0][15]
               + data->lbfluid[0][16] + data->lbfluid[0][17]
               + data->lbfluid[0][18]);
          // clang-format on
        }
        dens_ptr =
            (lb_float *)sc_array_index(density_values, mesh_iter->current_qid);
        *dens_ptr = dens;
#else  // LB_ADAPTIVE_GPU
        int patch_count = 0;
        dens_ptr = (double *)sc_array_index(
            density_values, cells_per_patch * mesh_iter->current_qid);
        for (int patch_z = 1; patch_z <= LBADAPT_PATCHSIZE; ++patch_z) {
          for (int patch_y = 1; patch_y <= LBADAPT_PATCHSIZE; ++patch_y) {
            for (int patch_x = 1; patch_x <= LBADAPT_PATCHSIZE; ++patch_x) {
              if (data->patch[patch_x][patch_y][patch_z].boundary) {
                dens = 0;
              } else {
                // clang-format off
                dens = avg_rho
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 0]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 1]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 2]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 3]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 4]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 5]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 6]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 7]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 8]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][ 9]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][10]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][11]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][12]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][13]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][14]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][15]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][16]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][17]
                     + data->patch[patch_x][patch_y][patch_z].lbfluid[0][18];
                // clang-format on
              }
              dens_ptr[patch_count] = dens;
              ++patch_count;
            }
          }
        }
#endif // LB_ADAPTIVE_GPU
      }
    }
  }
}

void lbadapt_get_velocity_values(sc_array_t *velocity_values) {
  const auto &forest = p4est_utils_get_forest_info(forest_order::adaptive_LB);
  int status;
  int level;
  double *veloc_ptr;
  lbadapt_payload_t *data;

#ifdef LB_ADAPTIVE_GPU
  int cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
#endif // LB_ADAPTIVE_GPU
  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  castable_unique_ptr<p4est_meshiter_t> mesh_iter;
  for (level = forest.coarsest_level_local; level <= forest.finest_level_local;
       ++level) {
    status = 0;
    mesh_iter.reset(p8est_meshiter_new_ext(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
        adapt_ghost->btype, P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER));

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[level].at(
            p8est_meshiter_get_current_storage_id(mesh_iter));

        /* calculate values to write */
        double rho;
        double j[3];

#ifndef LB_ADAPTIVE_GPU
        lbadapt_calc_local_fields(data->lbfluid, data->lbfields.force_density,
                                  data->lbfields.boundary,
                                  data->lbfields.has_force_density,
                                  p4est_params.h[level], &rho, j, nullptr);

        j[0] = j[0] / rho * h_max / lbpar.tau;
        j[1] = j[1] / rho * h_max / lbpar.tau;
        j[2] = j[2] / rho * h_max / lbpar.tau;

        veloc_ptr = (lb_float *)sc_array_index(
            velocity_values, P8EST_DIM * mesh_iter->current_qid);

        /* pass it into solution vector */
        std::memcpy(veloc_ptr, j, P8EST_DIM * sizeof(double));
#else  // LB_ADAPTIVE_GPU
        int patch_count = 0;
        for (int patch_z = 1; patch_z <= LBADAPT_PATCHSIZE; ++patch_z) {
          for (int patch_y = 1; patch_y <= LBADAPT_PATCHSIZE; ++patch_y) {
            for (int patch_x = 1; patch_x <= LBADAPT_PATCHSIZE; ++patch_x) {
              lb_float tmp_rho, tmp_j[3];
              lbadapt_calc_local_fields(
                  data->patch[patch_x][patch_y][patch_z].lbfluid,
                  data->patch[patch_x][patch_y][patch_z].force,
                  data->patch[patch_x][patch_y][patch_z].boundary, 1, h,
                  &tmp_rho, tmp_j, nullptr);

              rho = tmp_rho;
              j[0] = tmp_j[0] / rho * h_max / lbpar.tau;
              j[1] = tmp_j[1] / rho * h_max / lbpar.tau;
              j[2] = tmp_j[2] / rho * h_max / lbpar.tau;

              veloc_ptr = (double *)sc_array_index(
                  velocity_values,
                  P8EST_DIM *
                      (patch_count + cells_per_patch * mesh_iter->current_qid));

              /* pass it into solution vector */
              std::memcpy(veloc_ptr, j, P8EST_DIM * sizeof(double));

              ++patch_count;
            }
          }
        }
#endif // LB_ADAPTIVE_GPU
      }
    }
  }
}

// inefficient implementation (i.e. without dynamic programming)
void lbadapt_get_velocity_values_nodes(sc_array_t *velocity_values) {
  // FIXME Port to GPU
#ifndef LB_ADAPTIVE_GPU
  const auto &forest = p4est_utils_get_forest_info(forest_order::adaptive_LB);
  int status;
  int level;
  double rho;
  double vels[P4EST_DIM * P4EST_CHILDREN];
  double *veloc_ptr;
  lbadapt_payload_t *data;
  castable_unique_ptr<p4est_meshiter_t> m;
  bool smaller, larger;

  // neighbor search directions
  static const int n_idx[8][7] = {
      {0, 2, 14, 4, 10, 6, 18},  // left, front, bottom
      {1, 2, 15, 4, 11, 6, 19},  // right, front, bottom
      {0, 3, 16, 4, 10, 7, 20},  // left, back, bottom
      {1, 3, 17, 4, 11, 7, 21},  // right, back, bottom
      {0, 2, 14, 5, 12, 8, 22},  // left, front, top
      {1, 2, 15, 5, 13, 8, 23},  // right, front, top
      {0, 3, 16, 5, 12, 9, 24},  // left, back, top
      {1, 3, 17, 5, 13, 9, 25}}; // right, back, top

  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  for (level = forest.coarsest_level_local; level <= forest.finest_level_local;
       ++level) {
    // local mesh width
    lb_float h =
        (lb_float)P8EST_QUADRANT_LEN(level) / ((lb_float)P8EST_ROOT_LEN);

    m.reset(p4est_meshiter_new_ext(adapt_p4est, adapt_ghost, adapt_mesh,
                                   adapt_virtual, level, adapt_ghost->btype,
                                   P4EST_TRAVERSE_LOCAL, P4EST_TRAVERSE_REAL,
                                   P4EST_TRAVERSE_PARBOUNDINNER));
    status = 0;
    while (status != P4EST_MESHITER_DONE) {
      status = p4est_meshiter_next(m);
      if (status != P4EST_MESHITER_DONE) {
        // collect 8 velocity values
        for (int c = 0; c < P4EST_CHILDREN; ++c) {
          data = &lbadapt_local_data[level].at(
              p8est_meshiter_get_current_storage_id(m));
          lbadapt_calc_local_fields(data->lbfluid, data->lbfields.force_density,
                                    data->lbfields.boundary,
                                    data->lbfields.has_force_density, h, &rho,
                                    vels, nullptr);

          if (!data->lbfields.boundary) {
            vels[0] = vels[0] / rho * h_max / lbpar.tau;
            vels[1] = vels[1] / rho * h_max / lbpar.tau;
            vels[2] = vels[2] / rho * h_max / lbpar.tau;
          }

          for (int idx = 0; idx < 7; ++idx) {
            smaller = larger = false;
            p4est_meshiter_set_neighbor_quad_info(m, n_idx[c][idx]);
            if (m->neighbor_vid != -1) {
              larger = true;
            }

            // catch that neighbor may be smaller if nothing is found
            if (m->neighbor_qid == -1) {
              // need to set qid, vid, and is_ghost
              p4est_virtual_get_neighbor(adapt_p4est, adapt_ghost, adapt_mesh,
                                         adapt_virtual, m->current_qid, c,
                                         n_idx[c][idx], &m->neighbor_encs,
                                         &m->neighbor_qids, &m->neighbor_vids);
              P4EST_ASSERT(0 == m->neighbor_qids.elem_count ||
                           1 == m->neighbor_qids.elem_count);
              for (size_t i = 0; i < m->neighbor_qids.elem_count; ++i) {
                smaller = true;
                m->neighbor_qid = *(int *)sc_array_index(&m->neighbor_qids, i);
                if (adapt_p4est->local_num_quadrants <= m->neighbor_qid) {
                  m->neighbor_qid -= adapt_p4est->local_num_quadrants;
                  m->neighbor_is_ghost = 1;
                } else {
                  m->neighbor_is_ghost = 0;
                }
                m->neighbor_vid = -1;
              }
            }
            if (!smaller && !larger) {
              if (!m->neighbor_is_ghost) {
                data = &lbadapt_local_data[level].at(
                    p8est_meshiter_get_neighbor_storage_id(m));
              } else {
                data = &lbadapt_ghost_data[level].at(
                    p8est_meshiter_get_neighbor_storage_id(m));
              }
            } else if (smaller && !larger) {
              if (!m->neighbor_is_ghost) {
                data = &lbadapt_local_data[level + 1].at(
                    adapt_virtual->quad_qreal_offset[m->neighbor_qid]);
              } else {
                data = &lbadapt_ghost_data[level + 1].at(
                    adapt_virtual->quad_greal_offset[m->neighbor_qid]);
              }
            } else if (!smaller && larger) {
              if (!m->neighbor_is_ghost) {
                data = &lbadapt_local_data[level - 1].at(
                    adapt_virtual->quad_qreal_offset[m->neighbor_qid]);
              } else {
                data = &lbadapt_ghost_data[level - 1].at(
                    adapt_virtual->quad_greal_offset[m->neighbor_qid]);
              }
            } else {
              assert(false);
            }
            lbadapt_calc_local_fields(
                data->lbfluid, data->lbfields.force_density,
                data->lbfields.boundary, data->lbfields.has_force_density, h,
                &rho, &vels[P4EST_DIM * (idx + 1)], nullptr);

            if (!data->lbfields.boundary) {
              vels[P4EST_DIM * (idx + 1) + 0] =
                  vels[P4EST_DIM * (idx + 1) + 0] / rho * h_max / lbpar.tau;
              vels[P4EST_DIM * (idx + 1) + 1] =
                  vels[P4EST_DIM * (idx + 1) + 1] / rho * h_max / lbpar.tau;
              vels[P4EST_DIM * (idx + 1) + 2] =
                  vels[P4EST_DIM * (idx + 1) + 2] / rho * h_max / lbpar.tau;
            }
          }
          // perform trilinear interpolation ignoring the interpolation error
          // introduced by non-regular grid
          // x - direction
          for (int i = 0; i < P4EST_HALF; ++i) {
            vels[i * P4EST_DIM + 0] = 0.5 * (vels[2 * i * P4EST_DIM + 0] +
                                             vels[(2 * i + 1) * P4EST_DIM + 0]);
            vels[i * P4EST_DIM + 1] = 0.5 * (vels[2 * i * P4EST_DIM + 1] +
                                             vels[(2 * i + 1) * P4EST_DIM + 1]);
            vels[i * P4EST_DIM + 2] = 0.5 * (vels[2 * i * P4EST_DIM + 2] +
                                             vels[(2 * i + 1) * P4EST_DIM + 2]);
          }
          // y - direction
          for (int i = 0; i < 2; ++i) {
            vels[i * P4EST_DIM + 0] = 0.5 * (vels[2 * i * P4EST_DIM + 0] +
                                             vels[(2 * i + 1) * P4EST_DIM + 0]);
            vels[i * P4EST_DIM + 1] = 0.5 * (vels[2 * i * P4EST_DIM + 1] +
                                             vels[(2 * i + 1) * P4EST_DIM + 1]);
            vels[i * P4EST_DIM + 2] = 0.5 * (vels[2 * i * P4EST_DIM + 2] +
                                             vels[(2 * i + 1) * P4EST_DIM + 2]);
          }
          // z - direction
          vels[0] = 0.5 * (vels[0] + vels[P4EST_DIM + 0]);
          vels[1] = 0.5 * (vels[1] + vels[P4EST_DIM + 1]);
          vels[2] = 0.5 * (vels[2] + vels[P4EST_DIM + 2]);

          // write velocity to result array
          veloc_ptr = (double *)sc_array_index(
              velocity_values,
              (P4EST_CHILDREN * P4EST_DIM * m->current_qid) + (P4EST_DIM * c));
          std::memcpy(veloc_ptr, vels, P8EST_DIM * sizeof(double));
        }
      }
    }
  }
#endif // LB_ADAPTIVE_GPU
}

/** Check if velocity for a given quadrant is set and if not set it
 * @param[in]       qid        Quadrant whose velocity is needed
 * @param[in]       data       Numerical payload of quadrant
 * @param[in, out]  vel        Vector of velocities that are already known
 */
void check_vel(int qid, double local_h, lbadapt_payload_t *data,
               std::vector<std::array<double, 3>> &vel) {
#ifndef LB_ADAPTIVE_GPU
  double rho;
  if (vel[qid] == std::array<double, 3>{{std::numeric_limits<double>::min(),
                                         std::numeric_limits<double>::min(),
                                         std::numeric_limits<double>::min()}}) {
    lbadapt_calc_local_fields(data->lbfluid, data->lbfields.force_density,
                              data->lbfields.boundary,
                              data->lbfields.has_force_density, local_h, &rho,
                              vel[qid].data(), nullptr);

    lb_float h_max = p4est_params.h[p4est_params.max_ref_level];
    vel[qid][0] = vel[qid][0] / rho * h_max / lbpar.tau;
    vel[qid][1] = vel[qid][1] / rho * h_max / lbpar.tau;
    vel[qid][2] = vel[qid][2] / rho * h_max / lbpar.tau;
  }
#endif //! LB_ADAPTIVE_GPU
}

void lbadapt_get_vorticity_values(sc_array_t *vort_values) {
#ifndef LB_ADAPTIVE_GPU
  // FIXME port to GPU
  P4EST_ASSERT(vort_values->elem_count ==
               (size_t)P4EST_DIM * adapt_p4est->local_num_quadrants);

  // use dynamic programming to calculate fluid velocities
  std::vector<std::array<double, 3>> fluid_vel(
      adapt_p4est->local_num_quadrants + adapt_ghost->ghosts.elem_count,
      std::array<double, 3>({{std::numeric_limits<double>::min(),
                              std::numeric_limits<double>::min(),
                              std::numeric_limits<double>::min()}}));

  int nq;
  int c_level;
  p4est_locidx_t lq = adapt_p4est->local_num_quadrants;
  const std::array<int, 3> neighbor_dirs = std::array<int, 3>({{1, 3, 5}});
  castable_unique_ptr<sc_array_t> neighbor_qids = sc_array_new(sizeof(int));
  castable_unique_ptr<sc_array_t> neighbor_encs = sc_array_new(sizeof(int));
  std::array<std::vector<int>, 3> n_qids;
  std::array<double, 3> mesh_width{};
  double c_h, h;
  p8est_quadrant_t *quad;
  lbadapt_payload_t *data, *currCellData;
  double *ins;
  double vort[3];

  for (int qid = 0; qid < adapt_p4est->local_num_quadrants; ++qid) {
    quad = p8est_mesh_get_quadrant(adapt_p4est, adapt_mesh, qid);
    c_level = quad->level;
    currCellData =
        &lbadapt_local_data[c_level].at(adapt_virtual->quad_qreal_offset[qid]);
    c_h = (double)P8EST_QUADRANT_LEN(c_level) / (double)P8EST_ROOT_LEN;
    check_vel(qid, c_h, currCellData, fluid_vel);

    // get neighboring quads, their mesh width, and their velocity
    for (unsigned int dir_idx = 0; dir_idx < neighbor_dirs.size(); ++dir_idx) {
      sc_array_truncate(neighbor_qids);
      sc_array_truncate(neighbor_encs);

      p4est_mesh_get_neighbors(adapt_p4est, adapt_ghost, adapt_mesh, qid,
                               neighbor_dirs[dir_idx], nullptr, neighbor_encs,
                               neighbor_qids);
      P4EST_ASSERT(0 <= neighbor_qids->elem_count &&
                   neighbor_qids->elem_count <= P8EST_HALF);
      for (size_t i = 0; i < neighbor_qids->elem_count; ++i) {
        nq = *(int *)sc_array_index(neighbor_qids, i);
        bool is_ghost = (lq <= nq);
        if (!is_ghost) {
          quad = p8est_mesh_get_quadrant(adapt_p4est, adapt_mesh, nq);
          data = &lbadapt_local_data[quad->level].at(
              adapt_virtual->quad_qreal_offset[nq]);
        } else {
          quad = p4est_quadrant_array_index(&adapt_ghost->ghosts, nq - lq);
          data = &lbadapt_ghost_data[quad->level].at(
              adapt_virtual->quad_greal_offset[nq - lq]);
        }
        h = (double)P8EST_QUADRANT_LEN(quad->level) / (double)P8EST_ROOT_LEN;
        check_vel(nq, h, data, fluid_vel);
        n_qids[dir_idx].push_back(nq);
      }
      mesh_width[dir_idx] =
          0.5 *
          (((double)P8EST_QUADRANT_LEN(c_level) / (double)P8EST_ROOT_LEN) +
           ((double)P8EST_QUADRANT_LEN(quad->level) / (double)P8EST_ROOT_LEN));
    }

    // calculate vorticity from neighboring quadrants' velocities
    vort[0] = 0;
    vort[1] = 0;
    vort[2] = 0;
    if (!currCellData->lbfields.boundary) {
      for (size_t i = 0; i < n_qids[0].size(); ++i) {
        vort[0] += ((1. / n_qids[0].size()) *
                    (((fluid_vel[n_qids[0][i]][2] - fluid_vel[qid][2]) /
                      mesh_width[1]) -
                     ((fluid_vel[n_qids[0][i]][1] - fluid_vel[qid][1]) /
                      mesh_width[2])));
      }
      for (size_t i = 0; i < n_qids[1].size(); ++i) {
        vort[1] += ((1. / n_qids[1].size()) *
                    (((fluid_vel[n_qids[1][i]][0] - fluid_vel[qid][0]) /
                      mesh_width[2]) -
                     ((fluid_vel[n_qids[1][i]][2] - fluid_vel[qid][2]) /
                      mesh_width[0])));
      }
      for (size_t i = 0; i < n_qids[2].size(); ++i) {
        vort[2] += ((1. / n_qids[2].size()) *
                    (((fluid_vel[n_qids[2][i]][1] - fluid_vel[qid][1]) /
                      mesh_width[0]) -
                     ((fluid_vel[n_qids[2][i]][0] - fluid_vel[qid][0]) /
                      mesh_width[1])));
      }
    }
    // clear neighbor lists and copy result into array
    n_qids[0].clear();
    n_qids[1].clear();
    n_qids[2].clear();

    ins = (double *)sc_array_index(vort_values, 3 * qid);
    std::memcpy(ins, vort, 3 * sizeof(double));
  }
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_get_boundary_status() {
  int status;
  int level;
  lbadapt_payload_t *data;

  /* set boundary status */
  const auto &forest = p4est_utils_get_forest_info(forest_order::adaptive_LB);

  /** prepare exchanging boundary values */
  std::vector<lbadapt_payload_t *> local_pointer(P8EST_QMAXLEVEL);
  std::vector<lbadapt_payload_t *> ghost_pointer(P8EST_QMAXLEVEL);
  prepare_ghost_exchange(lbadapt_local_data, local_pointer, lbadapt_ghost_data,
                         ghost_pointer);
  castable_unique_ptr<p4est_meshiter_t> mesh_iter;

  for (level = forest.coarsest_level_global;
       level <= forest.finest_level_global; ++level) {
#ifdef LB_ADAPTIVE_GPU
    int base = P8EST_QUADRANT_LEN(level);
    int root = P8EST_ROOT_LEN;
    double patch_offset =
        ((lb_float)base / (LBADAPT_PATCHSIZE * (lb_float)root)) * 0.5;
    lb_float xyz_quad[3];
    double xyz_patch[3];
#endif // LB_ADAPTIVE_GPU

    status = 0;
    mesh_iter.reset(p4est_meshiter_new_ext(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, level,
        adapt_ghost->btype, P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER));

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        assert(!mesh_iter->current_is_ghost);
        data = &lbadapt_local_data[level].at(
            p8est_meshiter_get_current_storage_id(mesh_iter));

#ifndef LB_ADAPTIVE_GPU
        double midpoint[3];
        p4est_utils_get_midpoint(mesh_iter, midpoint);

        data->lbfields.boundary = lbadapt_is_boundary(midpoint);
#else  // LB_ADAPTIVE_GPU
        p4est_utils_get_front_lower_left(mesh_iter, (double *)xyz_quad);
        for (int i = 0; i < 3; ++i) {
          xyz_quad[i] *= box_l[i] / lb_conn_brick[i];
        }
        bool all_boundary = true;

        for (int patch_z = 0; patch_z < LBADAPT_PATCHSIZE; ++patch_z) {
          for (int patch_y = 0; patch_y < LBADAPT_PATCHSIZE; ++patch_y) {
            for (int patch_x = 0; patch_x < LBADAPT_PATCHSIZE; ++patch_x) {
              xyz_patch[0] =
                  xyz_quad[0] + 2 * patch_x * patch_offset + patch_offset;
              xyz_patch[1] =
                  xyz_quad[1] + 2 * patch_y * patch_offset + patch_offset;
              xyz_patch[2] =
                  xyz_quad[2] + 2 * patch_z * patch_offset + patch_offset;
              data->patch[1 + patch_x][1 + patch_y][1 + patch_z].boundary =
                  lbadapt_is_boundary(xyz_patch);
              all_boundary =
                  all_boundary &&
                  data->patch[1 + patch_x][1 + patch_y][1 + patch_z].boundary;
            }
          }
        }
        data->boundary = all_boundary;
#endif // LB_ADAPTIVE_GPU
      }
    }

    p4est_virtual_ghost_exchange_data_level(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual,
        adapt_virtual_ghost, level, sizeof(lbadapt_payload_t),
        (void **)local_pointer.data(), (void **)ghost_pointer.data());
  }
}

void lbadapt_calc_local_rho(p8est_meshiter_t *mesh_iter, lb_float *rho) {
#ifndef LB_ADAPTIVE_GPU
  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  lbadapt_payload_t *data;
  data = &lbadapt_local_data[mesh_iter->current_level].at(
      p8est_meshiter_get_neighbor_storage_id(mesh_iter));

  lb_float avg_rho = lbpar.rho * h_max * h_max * h_max;

  // clang-format off
  *rho += avg_rho +
          data->lbfluid[0][ 0] + data->lbfluid[0][ 1] + data->lbfluid[0][ 2] +
          data->lbfluid[0][ 3] + data->lbfluid[0][ 4] + data->lbfluid[0][ 5] +
          data->lbfluid[0][ 6] + data->lbfluid[0][ 7] + data->lbfluid[0][ 8] +
          data->lbfluid[0][ 9] + data->lbfluid[0][10] + data->lbfluid[0][11] +
          data->lbfluid[0][12] + data->lbfluid[0][13] + data->lbfluid[0][14] +
          data->lbfluid[0][15] + data->lbfluid[0][16] + data->lbfluid[0][17] +
          data->lbfluid[0][18];
// clang-format on
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_calc_local_j(p8est_meshiter_t *mesh_iter, lb_float *j) {
#ifndef LB_ADAPTIVE_GPU
  lbadapt_payload_t *data;
  data = &lbadapt_local_data[mesh_iter->current_level].at(
      p8est_meshiter_get_neighbor_storage_id(mesh_iter));

  // clang-format off
  j[0] = data->lbfluid[0][ 1] - data->lbfluid[0][ 2] + data->lbfluid[0][ 7] -
         data->lbfluid[0][ 8] + data->lbfluid[0][ 9] - data->lbfluid[0][10] +
         data->lbfluid[0][11] - data->lbfluid[0][12] + data->lbfluid[0][13] -
         data->lbfluid[0][14];
  j[1] = data->lbfluid[0][ 3] - data->lbfluid[0][ 4] + data->lbfluid[0][ 7] -
         data->lbfluid[0][ 8] - data->lbfluid[0][ 9] + data->lbfluid[0][10] +
         data->lbfluid[0][15] - data->lbfluid[0][16] + data->lbfluid[0][17] -
         data->lbfluid[0][18];
  j[2] = data->lbfluid[0][ 5] - data->lbfluid[0][ 6] + data->lbfluid[0][11] -
         data->lbfluid[0][12] - data->lbfluid[0][13] + data->lbfluid[0][14] +
         data->lbfluid[0][15] - data->lbfluid[0][16] - data->lbfluid[0][17] +
         data->lbfluid[0][18];
// clang-format on
#endif // LB_ADAPTIVE_GPU
}

/*** ITERATOR CALLBACKS ***/
void lbadapt_calc_local_rho(p8est_iter_volume_info_t *info, void *user_data) {
#ifndef LB_ADAPTIVE_GPU
  auto *rho = (lb_float *)user_data; /* passed lb_float to fill */
  p8est_quadrant_t *q = info->quad;
  p4est_locidx_t qid = info->quadid;
  lbadapt_payload_t *data =
      &lbadapt_local_data[q->level].at(adapt_virtual->quad_qreal_offset[qid]);
  lb_float h_max = p4est_params.h[p4est_params.max_ref_level];

  // unit conversion: mass density
  if (!(lattice_switch & LATTICE_LB)) {
    runtimeErrorMsg() << "Error in lb_calc_local_rho in " << __FILE__
                      << __LINE__ << ": CPU LB not switched on.";
    *rho = 0;
    return;
  }

  lb_float avg_rho = lbpar.rho * h_max * h_max * h_max;

  // clang-format off
  *rho += avg_rho +
          data->lbfluid[0][ 0] + data->lbfluid[0][ 1] + data->lbfluid[0][ 2] +
          data->lbfluid[0][ 3] + data->lbfluid[0][ 4] + data->lbfluid[0][ 5] +
          data->lbfluid[0][ 6] + data->lbfluid[0][ 7] + data->lbfluid[0][ 8] +
          data->lbfluid[0][ 9] + data->lbfluid[0][10] + data->lbfluid[0][11] +
          data->lbfluid[0][12] + data->lbfluid[0][13] + data->lbfluid[0][14] +
          data->lbfluid[0][15] + data->lbfluid[0][16] + data->lbfluid[0][17] +
          data->lbfluid[0][18];
// clang-format on
#endif // LB_ADAPTIVE_GPU
}

void calc_local_j(lb_float populations[2][19], std::array<lb_float, 3> &res) {
  // clang-format off
  res[0] = populations[0][ 1] - populations[0][ 2] + populations[0][ 7] -
           populations[0][ 8] + populations[0][ 9] - populations[0][10] +
           populations[0][11] - populations[0][12] + populations[0][13] -
           populations[0][14];
  res[1] = populations[0][ 3] - populations[0][ 4] + populations[0][ 7] -
           populations[0][ 8] - populations[0][ 9] + populations[0][10] +
           populations[0][15] - populations[0][16] + populations[0][17] -
           populations[0][18];
  res[2] = populations[0][ 5] - populations[0][ 6] + populations[0][11] -
           populations[0][12] - populations[0][13] + populations[0][14] +
           populations[0][15] - populations[0][16] - populations[0][17] +
           populations[0][18];
  // clang-format on
}
void lbadapt_calc_fluid_momentum(p8est_iter_volume_info_t *info,
                                 void *user_data) {
#ifndef LB_ADAPTIVE_GPU
  auto *momentum = (double *)user_data;

  p8est_quadrant_t *q = info->quad;
  p4est_locidx_t qid = info->quadid;
  lbadapt_payload_t *data;
  data =
      &lbadapt_local_data[q->level].at(adapt_virtual->quad_qreal_offset[qid]);

  std::array<lb_float, 3> j{};
  calc_local_j(data->lbfluid, j);
  for (int i = 0; i < P8EST_DIM; ++i)
    momentum[i] += j[i] + data->lbfields.force_density[i];
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_calc_local_temp(p8est_iter_volume_info_t *info, void *user_data) {
#ifndef LB_ADAPTIVE_GPU
  auto *ti = (temp_iter_t *)user_data;

  p8est_quadrant_t *q = info->quad;
  p4est_locidx_t qid = info->quadid;
  lbadapt_payload_t *data;
  data =
      &lbadapt_local_data[q->level].at(adapt_virtual->quad_qreal_offset[qid]);
  std::array<lb_float, 3> j{};

  if (data->lbfields.boundary) {
    j = {{0., 0., 0.}};
    ++ti->n_non_boundary_nodes;
  } else {
    calc_local_j(data->lbfluid, j);
  }
  ti->temp += scalar(j.data(), j.data());

#endif // LB_ADAPTIVE_GPU
}

void lbadapt_dump2file_synced(std::string &filename) {
#ifndef LB_ADAPTIVE_GPU
  int nqid = 0;
  p4est_quadrant_t *q;

  for (int qid = 0; qid < adapt_p4est->global_num_quadrants; ++qid) {
    // Synchronization point
    MPI_Barrier(adapt_p4est->mpicomm);

    // MPI rank holding current quadrant will open the file, append its
    // information, flush it, and close the file afterwards.
    if ((adapt_p4est->global_first_quadrant[adapt_p4est->mpirank] <= qid) &&
        (qid < adapt_p4est->global_first_quadrant[adapt_p4est->mpirank + 1])) {
      // get quadrant for level information
      q = p4est_mesh_get_quadrant(adapt_p4est, adapt_mesh, nqid);
      // fetch payload
      lbadapt_payload_t *data = &lbadapt_local_data[q->level].at(
          adapt_virtual->quad_qreal_offset[nqid]);
      // we are not interested in boundary quadrants
      if (!data->lbfields.boundary) {
        std::ofstream myfile;
        myfile.open(filename, std::ofstream::out | std::ofstream::app);
        myfile << "id: " << qid << " level: " << (int)q->level << std::endl
#if 0
               << " is parallel boundary: "
               << (-1 != adapt_mesh->parallel_boundary[nqid])
#endif // 0
               << "; has virtuals: "
               << (-1 != adapt_virtual->virtual_qflags[nqid]) << std::endl
               << " - distributions: " << std::endl
               << "0: ";
        for (int i = 0; i < 19; ++i) {
          myfile << data->lbfluid[0][i] << " - ";
        }
        myfile << std::endl << "1: ";
        for (int i = 0; i < 19; ++i) {
          myfile << data->lbfluid[1][i] << " - ";
        }
        myfile << std::endl;
#ifdef DUMP_VIRTUALS
        if (-1 != adapt_virtual->virtual_qflags[nqid]) {
          for (int v = 0; v < P4EST_CHILDREN; ++v) {
            data = &lbadapt_local_data[q->level + 1].at(
                adapt_virtual->quad_qvirtual_offset[nqid] + v);
            myfile << "virtual distributions: v" << v << std::endl << "0: ";
            for (int i = 0; i < 19; ++i) {
              myfile << data->lbfluid[0][i] << " - ";
            }
            myfile << std::endl << "1: ";
            for (int i = 0; i < 19; ++i) {
              myfile << data->lbfluid[1][i] << " - ";
            }
            myfile << std::endl;
          }
        }
#endif // DUMP_VIRTUALS
        myfile << std::endl;

        myfile.flush();
        myfile.close();
      }
      // increment local quadrant index
      ++nqid;
    }
  }
  // make sure that we have inspected all local quadrants.
  P4EST_ASSERT(nqid == adapt_p4est->local_num_quadrants);
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_dump2file(p8est_iter_volume_info_t *info, void *user_data) {
#ifndef LB_ADAPTIVE_GPU
  p8est_quadrant_t *q = info->quad;
  lbadapt_payload_t *data = &lbadapt_local_data[q->level].at(
      adapt_virtual->quad_qreal_offset[info->quadid]);

  auto *filename = (std::string *)user_data;
  std::ofstream myfile;
  myfile.open(*filename, std::ofstream::out | std::ofstream::app);
  myfile << "id: "
         << info->quadid +
                adapt_p4est->global_first_quadrant[adapt_p4est->mpirank]
         << "; coords: " << (q->x / (1 << (P8EST_MAXLEVEL - q->level))) << ", "
         << (q->y / (1 << (P8EST_MAXLEVEL - q->level))) << ", "
         << (q->z / (1 << (P8EST_MAXLEVEL - q->level)))
         << "; boundary: " << data->lbfields.boundary << std::endl
         << " - distributions: " << std::endl;
  for (int i = 0; i < 19; ++i) {
    myfile << data->lbfluid[0][i] << " - ";
  }
  myfile << std::endl;
  for (int i = 0; i < 19; ++i) {
    myfile << data->lbfluid[1][i] << " - ";
  }
  myfile << std::endl << std::endl;

  myfile.flush();
  myfile.close();
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_init_qid_payload(p8est_iter_volume_info_t *info, void *user_data) {
  p8est_quadrant_t *q = info->quad;
  p8est_tree_t *t = p4est_tree_array_index(info->p4est->trees, info->treeid);
  q->p.user_long = info->quadid + t->quadrants_offset;
}

void lbadapt_interpolate_pos_adapt(const Vector3d &opos,
                                   std::vector<lbadapt_payload_t *> &payloads,
                                   std::vector<double> &interpol_weights,
                                   std::vector<int> &levels) {
  P4EST_ASSERT(payloads.empty() && interpol_weights.empty() && levels.empty());

  const auto &forest = p4est_utils_get_forest_info(forest_order::adaptive_LB);
  int nearest_corner = 0; // This value determines first index of
                          // neighbor_directions and weight_indices.
  uint64_t quad_index;
  // Order neighbor_directions as follows:
  // face neighbor x, face neighbor y, edge neighbor xy plane,
  // face neighbor z, edge neighbor xz plane, edge neighbor yz plane,
  // corner neighbor.
  // The reason why this order is chosen is that we can now XOR the
  // nearest_corner index to determine the weight indices.
  const std::array<std::array<int, 7>, 8> neighbor_directions = {{
      {{0, 2, 14, 4, 10, 6, 18}}, // left, front, bottom
      {{1, 2, 15, 4, 11, 6, 19}}, // right, front, bottom
      {{0, 3, 16, 4, 10, 7, 20}}, // left, back, bottom
      {{1, 3, 17, 4, 11, 7, 21}}, // right, back, bottom
      {{0, 2, 14, 5, 12, 8, 22}}, // left, front, top
      {{1, 2, 15, 5, 13, 8, 23}}, // right, front, top
      {{0, 3, 16, 5, 12, 9, 24}}, // left, back, top
      {{1, 3, 17, 5, 13, 9, 25}}, // right, back, top
  }};

  std::array<double, 6>
      interpolation_weights{}; // Stores normed distance from pos to center
                               // of cell containing pos in both directions.
                               // order: lower x, lower y, lower z,
                               //        upper x, upper y, upper z
  const std::array<std::array<int, 3>, 8> weight_indices = {{
      // indexes into interpolation_weights
      {{0, 1, 2}},
      {{3, 1, 2}},
      {{0, 4, 2}},
      {{3, 4, 2}},
      {{0, 1, 5}},
      {{3, 1, 5}},
      {{0, 4, 5}},
      {{3, 4, 5}},
  }};

  std::array<int, 3> fold = {{0, 0, 0}};
  std::array<double, 3> pos = {{opos[0], opos[1], opos[2]}};
  fold_position(pos, fold);

  int64_t qidx = p4est_utils_pos_to_qid(forest_order::adaptive_LB, pos.data());
  if (!(0 <= qidx && qidx < adapt_p4est->local_num_quadrants)) {
    lbadapt_interpolate_pos_ghost(opos, payloads, interpol_weights, levels);
    if (!payloads.empty())
      return;

    std::stringstream err_msg;
    err_msg << "At sim time " << sim_time << " (lbsteps: " << n_lbsteps << ")"
            << std::endl
            << "Particle not in local LB domain of rank " << this_node
            << " Found qid " << qidx << " position " << pos[0] << ", " << pos[1]
            << ", " << pos[2] << " quad idx "
            << p4est_utils_pos_to_index(forest_order::adaptive_LB, pos.data())
            << "; LB process boundary indices:" << std::endl;
    for (int i = 0; i <= n_nodes; ++i) {
      err_msg << forest.p4est_space_idx[i] << " ";
    }
    err_msg << std::endl;
#ifdef DD_P4EST
    err_msg << "belongs to MD process "
            << cell_structure.position_to_node(pos.data()) << " ("
            << p4est_utils_pos_to_proc(forest_order::short_range, pos.data())
            << " in forestinfo)" << std::endl;
#endif
    err_msg << std::endl;
    fprintf(stderr, "%s", err_msg.str().c_str());
    errexit();
  } else {
    coupling_quads[qidx] = true;
    P4EST_ASSERT(p4est_utils_pos_sanity_check(qidx, pos.data()));
  }

  int lvl, sid;
  p8est_quadrant_t *quad, *q;
  quad = p8est_mesh_get_quadrant(adapt_p4est, adapt_mesh, qidx);
  lvl = quad->level;
  sid = adapt_virtual->quad_qreal_offset[qidx];
  payloads.push_back(&lbadapt_local_data[lvl].at(sid));
  levels.push_back(lvl);

  std::array<double, 3> quad_pos{}, neighbor_quad_pos{};
  double dis;
  p4est_topidx_t tree;
  p4est_utils_get_midpoint(adapt_p4est, adapt_mesh->quad_to_tree[qidx], quad,
                           quad_pos.data());
  for (int d = 0; d < 3; ++d) {
    dis = (pos[d] - quad_pos[d]) / p4est_params.h[lvl];
    P4EST_ASSERT(-0.5 <= dis && dis <= 0.5);
    if (dis > 0.0) { // right neighbor
      nearest_corner |= 1 << d;
      interpolation_weights[d] = dis;
      interpolation_weights[d + 3] = 1.0 - dis;
    } else {
      interpolation_weights[d + 3] = abs(dis);
      interpolation_weights[d] = 1.0 - interpolation_weights[d + 3];
    }
  }
  interpol_weights.push_back(
      interpolation_weights[weight_indices[nearest_corner][0]] *
      interpolation_weights[weight_indices[nearest_corner][1]] *
      interpolation_weights[weight_indices[nearest_corner][2]]);

  castable_unique_ptr<sc_array_t> nenc = sc_array_new(sizeof(int));
  castable_unique_ptr<sc_array_t> nqid = sc_array_new(sizeof(int));
  castable_unique_ptr<sc_array_t> nvid = sc_array_new(sizeof(int));
  p4est_locidx_t lq = adapt_mesh->local_num_quadrants;
  p4est_locidx_t gq = adapt_mesh->ghost_num_quadrants;
  int idx;
  for (int i = 0; i < 7; ++i) {
    p8est_virtual_get_neighbor(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual, qidx, -1,
        neighbor_directions[nearest_corner][i], nenc, nqid, nvid);
    // if neighbor is smaller we will not obtain any neighbors from virtual
    // neighbor query
    if (!nqid->elem_count) {
      p8est_mesh_get_neighbors(adapt_p4est, adapt_ghost, adapt_mesh, qidx,
                               neighbor_directions[nearest_corner][i], nullptr,
                               nullptr, nqid);
    }
    for (size_t n = 0; n < nqid->elem_count; ++n) {
      idx = *((int *)sc_array_index_int(nqid, n));

      if (0 <= idx && idx < lq) { // local quadrant
        coupling_quads[idx] = true;
        q = p8est_mesh_get_quadrant(adapt_p4est, adapt_mesh, idx);
        tree = adapt_mesh->quad_to_tree[idx];
        lvl = q->level;
        sid = adapt_virtual->quad_qreal_offset[idx];
        payloads.push_back(&lbadapt_local_data[lvl].at(sid));
      } else if (lq <= idx && idx < lq + gq) {
        idx -= lq;
        q = p8est_quadrant_array_index(&adapt_ghost->ghosts, idx);
        tree = adapt_mesh->ghost_to_tree[idx];
        lvl = q->level;
        sid = adapt_virtual->quad_greal_offset[idx];
        payloads.push_back(&lbadapt_ghost_data[lvl].at(sid));
      } else {
        SC_ABORT_NOT_REACHED();
      }

      p4est_utils_get_midpoint(adapt_p4est, tree, q, neighbor_quad_pos.data());
      std::array<double, 6> check_weights{};
      P4EST_ASSERT(p4est_utils_pos_vicinity_check(quad_pos, quad->level,
                                                  neighbor_quad_pos, q->level));
      P4EST_ASSERT(p4est_utils_pos_enclosing_check(quad_pos, quad->level,
                                                   neighbor_quad_pos, q->level,
                                                   pos, check_weights));
      P4EST_ASSERT(i != 7 || quad->level != q->level ||
                   ((std::abs(check_weights[0] - interpolation_weights[0]) <
                     ROUND_ERROR_PREC) &&
                    (std::abs(check_weights[1] - interpolation_weights[1]) <
                     ROUND_ERROR_PREC) &&
                    (std::abs(check_weights[2] - interpolation_weights[2]) <
                     ROUND_ERROR_PREC)));

      interpol_weights.push_back(
          interpolation_weights[weight_indices[nearest_corner ^ (i + 1)][0]] *
          interpolation_weights[weight_indices[nearest_corner ^ (i + 1)][1]] *
          interpolation_weights[weight_indices[nearest_corner ^ (i + 1)][2]] /
          (double)(nqid->elem_count));
      levels.push_back(lvl);
    }
    sc_array_truncate(nenc);
    sc_array_truncate(nqid);
    sc_array_truncate(nvid);
  }

  double dsum =
      std::accumulate(interpol_weights.begin(), interpol_weights.end(), 0.0);
  if (abs(1.0 - dsum) > ROUND_ERROR_PREC) {
    std::stringstream err_msg;
    err_msg << "[local interpol.] Sum of interpol. weights deviates from 1 by"
            << (1.0 - dsum) << std::endl;
    for (auto &w : interpol_weights) {
      err_msg << w << " ";
    }
    err_msg << std::endl;
    err_msg << "Rank " << this_node << " position " << pos[0] << ", " << pos[1]
            << ", " << pos[2] << " pos index"
            << p4est_utils_pos_to_index(forest_order::adaptive_LB, pos.data())
            << "LB process boundary indices:" << std::endl;
    for (int i = 0; i <= n_nodes; ++i) {
      err_msg << forest.p4est_space_idx[i] << " ";
    }
    err_msg << std::endl;
    err_msg << std::endl;
    fprintf(stderr, "%s", err_msg.str().c_str());
    errexit();
  }
}

void lbadapt_interpolate_pos_ghost(const Vector3d &opos,
                                   std::vector<lbadapt_payload_t *> &payloads,
                                   std::vector<double> &interpol_weights,
                                   std::vector<int> &levels) {
  P4EST_ASSERT(payloads.empty() && interpol_weights.empty() && levels.empty());
  const auto &forest = p4est_utils_get_forest_info(forest_order::adaptive_LB);
  int nearest_corner = 0;
  uint64_t quad_index;
  std::vector<p4est_locidx_t> quads_to_mark = {};

  std::array<double, 6>
      interpolation_weights{}; // Stores normed distance from pos to center
  // of cell containing pos in both directions.
  // order: lower x, lower y, lower z,
  //        upper x, upper y, upper z
  const std::array<std::array<int, 3>, 8> weight_indices = {{
      {{0, 1, 2}},
      {{3, 1, 2}},
      {{0, 4, 2}},
      {{3, 4, 2}},
      {{0, 1, 5}},
      {{3, 1, 5}},
      {{0, 4, 5}},
      {{3, 4, 5}},
  }};

  // Fold position.
  std::array<int, 3> fold = {{0, 0, 0}};
  std::array<double, 3> pos = {{opos[0], opos[1], opos[2]}};
  fold_position(pos, fold);

  // Find quadrant containing position and store its payload.
  // Begin by search in ghost layer.  If there is no quadrant containing the
  // given position search mirror quadrants.
  bool found_in_ghost = true;
  uint64_t pos_idx =
      p4est_utils_pos_to_index(forest_order::adaptive_LB, pos.data());
  std::vector<p4est_locidx_t> quad_indices;
  p4est_utils_bin_search_quad_in_array(pos_idx, &adapt_ghost->ghosts,
                                       quad_indices);
  if (quad_indices.empty() ||
      !p4est_utils_pos_sanity_check(quad_indices[0], pos.data(),
                                    found_in_ghost)) {
    quad_indices.clear();
    p4est_utils_bin_search_quad_in_array(pos_idx, &adapt_ghost->mirrors,
                                         quad_indices);
    if (!quad_indices.empty()) {
      found_in_ghost = false;
      quad_indices[0] = adapt_mesh->mirror_qid[quad_indices[0]];
    }
  }

  if (quad_indices.empty() ||
      !p4est_utils_pos_sanity_check(quad_indices[0], pos.data(),
                                    found_in_ghost)) {
    return;
  }

  int lvl, sid, tree, zsize;
  p8est_quadrant_t *quad;
  if (found_in_ghost) {
    quad = p8est_quadrant_array_index(&adapt_ghost->ghosts, quad_indices[0]);
    lvl = quad->level;
    tree = adapt_mesh->ghost_to_tree[quad_indices[0]];
    sid = adapt_virtual->quad_greal_offset[quad_indices[0]];
    payloads.push_back(&lbadapt_ghost_data[lvl].at(sid));
  } else {
    quad = p4est_mesh_get_quadrant(adapt_p4est, adapt_mesh, quad_indices[0]);
    quads_to_mark.push_back(quad_indices[0]);
    lvl = quad->level;
    tree = adapt_mesh->quad_to_tree[quad_indices[0]];
    sid = adapt_virtual->quad_qreal_offset[quad_indices[0]];
    payloads.push_back(&lbadapt_local_data[lvl].at(sid));
  }
  levels.push_back(lvl);

  // determine which neighbors to find
  std::array<double, 3> quad_pos{}, neighbor_quad_pos{};
  p4est_utils_get_midpoint(adapt_p4est, tree, quad, quad_pos.data());
  double dis;
  for (int d = 0; d < 3; ++d) {
    dis = (pos[d] - quad_pos[d]) / p4est_params.h[lvl];
    P4EST_ASSERT(-0.5 <= dis && dis <= 0.5);
    if (dis > 0.0) { // right neighbor
      nearest_corner |= 1 << d;
      interpolation_weights[d] = dis;
      interpolation_weights[d + 3] = 1.0 - dis;
    } else {
      interpolation_weights[d + 3] = abs(dis);
      interpolation_weights[d] = 1.0 - interpolation_weights[d + 3];
    }
  }
  interpol_weights.push_back(
      interpolation_weights[weight_indices[nearest_corner][0]] *
      interpolation_weights[weight_indices[nearest_corner][1]] *
      interpolation_weights[weight_indices[nearest_corner][2]]);
  zsize = 1 << (p4est_params.max_ref_level - lvl);

  // determine which neighbor indices need to be found
  std::array<uint64_t, 7> neighbor_indices = {
      {std::numeric_limits<uint64_t>::max(),
       std::numeric_limits<uint64_t>::max(),
       std::numeric_limits<uint64_t>::max(),
       std::numeric_limits<uint64_t>::max(),
       std::numeric_limits<uint64_t>::max(),
       std::numeric_limits<uint64_t>::max(),
       std::numeric_limits<uint64_t>::max()}};
  std::array<int, 3> displace{};
  for (int i = 1; i < 8; ++i) {
    // reset displace
    displace = {{0, 0, 0}};

    // set offset
    if (i & 1) {
      if (nearest_corner & 1)
        displace[0] = zsize;
      else
        displace[0] = -zsize;
    }
    if (i & 2) {
      if (nearest_corner & 2)
        displace[1] = zsize;
      else
        displace[1] = -zsize;
    }
    if (i & 4) {
      if (nearest_corner & 4)
        displace[2] = zsize;
      else
        displace[2] = -zsize;
    }
    P4EST_ASSERT(neighbor_indices[i - 1] ==
                 std::numeric_limits<uint64_t>::max());
    neighbor_indices[i - 1] =
        p4est_utils_global_idx(forest, quad, tree, displace);
  }

  // collect over all 7 direction wrt. to corner
  p8est_quadrant_t *q;
  int n_neighbors_per_dir;
  p4est_locidx_t mirror_qid;
  for (int dir = 1; dir < 8; ++dir) {
    n_neighbors_per_dir = 0;
    // search for neighbor in mirrors
    quad_indices.clear();
    p4est_utils_bin_search_quad_in_array(neighbor_indices[dir - 1],
                                         &adapt_ghost->mirrors, quad_indices,
                                         quad->level);
    for (int quad_indice : quad_indices) {
      mirror_qid = adapt_mesh->mirror_qid[quad_indice];
      sid = adapt_virtual->quad_qreal_offset[mirror_qid];
      q = p4est_mesh_get_quadrant(adapt_p4est, adapt_mesh, mirror_qid);
      P4EST_ASSERT((lvl - 1) <= q->level && q->level <= (lvl + 1));
      p4est_utils_get_midpoint(adapt_p4est, tree, q, neighbor_quad_pos.data());
      if (p4est_utils_pos_vicinity_check(quad_pos, quad->level,
                                         neighbor_quad_pos, q->level)) {
        payloads.push_back(&lbadapt_local_data[q->level].at(sid));
        levels.push_back(q->level);
        interpol_weights.push_back(
            interpolation_weights[weight_indices[nearest_corner ^ dir][0]] *
            interpolation_weights[weight_indices[nearest_corner ^ dir][1]] *
            interpolation_weights[weight_indices[nearest_corner ^ dir][2]]);
        quads_to_mark.push_back(mirror_qid);
        if (q->level > lvl) {
          if (dir == 1 || dir == 2 || dir == 4)
            interpol_weights[interpol_weights.size() - 1] *= 0.25;
          if (dir == 3 || dir == 5 || dir == 6)
            interpol_weights[interpol_weights.size() - 1] *= 0.5;
        }
        ++n_neighbors_per_dir;

        std::array<double, 6> check_weights{};
        P4EST_ASSERT(p4est_utils_pos_enclosing_check(
            quad_pos, quad->level, neighbor_quad_pos, q->level, pos,
            check_weights));
#if 0
        P4EST_ASSERT(dir != 7 || quad->level != q->level ||
                     ((std::abs(check_weights[0] - interpolation_weights[0]) <
                       ROUND_ERROR_PREC) &&
                      (std::abs(check_weights[1] - interpolation_weights[1]) <
                       ROUND_ERROR_PREC) &&
                      (std::abs(check_weights[2] - interpolation_weights[2]) <
                       ROUND_ERROR_PREC)));
#endif
      }
    }

    // search for neighbor in ghosts
    quad_indices.clear();
    p4est_utils_bin_search_quad_in_array(neighbor_indices[dir - 1],
                                         &adapt_ghost->ghosts, quad_indices,
                                         quad->level);
    for (int quad_indice : quad_indices) {
      sid = adapt_virtual->quad_greal_offset[quad_indice];
      q = p4est_quadrant_array_index(&adapt_ghost->ghosts, quad_indice);
      P4EST_ASSERT((lvl - 1) <= q->level && q->level <= (lvl + 1));
      p4est_utils_get_midpoint(adapt_p4est, tree, q, neighbor_quad_pos.data());
      if (p4est_utils_pos_vicinity_check(quad_pos, quad->level,
                                         neighbor_quad_pos, q->level)) {
        payloads.push_back(&lbadapt_ghost_data[q->level].at(sid));
        levels.push_back(q->level);
        interpol_weights.push_back(
            interpolation_weights[weight_indices[nearest_corner ^ dir][0]] *
            interpolation_weights[weight_indices[nearest_corner ^ dir][1]] *
            interpolation_weights[weight_indices[nearest_corner ^ dir][2]]);
        auto last_element = interpol_weights.back();
        if (q->level > lvl) {
          if (dir == 1 || dir == 2 || dir == 4)
            interpol_weights[interpol_weights.size() - 1] *= 0.25;
          if (dir == 3 || dir == 5 || dir == 6)
            interpol_weights[interpol_weights.size() - 1] *= 0.5;
        }
        ++n_neighbors_per_dir;

        std::array<double, 6> check_weights{};
        P4EST_ASSERT(p4est_utils_pos_enclosing_check(
            quad_pos, quad->level, neighbor_quad_pos, q->level, pos,
            check_weights));
        P4EST_ASSERT(dir != 7 || quad->level != q->level ||
                     ((std::abs(check_weights[0] - interpolation_weights[0]) <
                       ROUND_ERROR_PREC) &&
                      (std::abs(check_weights[1] - interpolation_weights[1]) <
                       ROUND_ERROR_PREC) &&
                      (std::abs(check_weights[2] - interpolation_weights[2]) <
                       ROUND_ERROR_PREC)));
      }
    }
    P4EST_ASSERT(payloads.size() <= 20);
    P4EST_ASSERT(n_neighbors_per_dir == 0 || n_neighbors_per_dir == 1 ||
                 n_neighbors_per_dir == 2 || n_neighbors_per_dir == 4);
    if (!n_neighbors_per_dir) {
      // not all neighbors for all directions exist => position is in outer halo
      payloads.clear();
      interpol_weights.clear();
      levels.clear();
      quads_to_mark.clear();
      return;
    }
  }

  for (p4est_locidx_t idx : quads_to_mark) {
    coupling_quads[idx] = true;
  }

  double dsum =
      std::accumulate(interpol_weights.begin(), interpol_weights.end(), 0.0);
  if (abs(1.0 - dsum) > ROUND_ERROR_PREC) {
    std::stringstream err_msg;
    err_msg << "[Ghost interpol. on rank " << this_node
            << "] Sum of interpol. weights deviates from 1 by " << (1.0 - dsum)
            << std::endl;
    for (auto &w : interpol_weights) {
      err_msg << w << " ";
    }
    err_msg << std::endl;
    err_msg << "Rank " << this_node << " position " << pos[0] << ", " << pos[1]
            << ", " << pos[2] << " pos index "
            << p4est_utils_pos_to_index(forest_order::adaptive_LB, pos.data())
            << " (found in ghost: " << found_in_ghost << ") "
            << "LB process boundary indices:" << std::endl;
    for (int i = 0; i <= n_nodes; ++i) {
      err_msg << forest.p4est_space_idx[i] << " ";
    }
    err_msg << std::endl;
    err_msg << std::endl;
    fprintf(stderr, "%s", err_msg.str().c_str());
  }
}

int lbadapt_sanity_check_parameters() {
  for (int level = p4est_params.min_ref_level;
       level <= p4est_params.max_ref_level; ++level) {
    if (abs(lbpar.gamma_shear[level]) > 1.0) {
      fprintf(stderr,
              "Bad relaxation parameter gamma_shear on level %i (%lf)\n", level,
              lbpar.gamma_shear[level]);
      errexit();
    }
    if (abs(lbpar.gamma_shear[level]) > 1.0) {
      fprintf(stderr, "Bad relaxation parameter gamma_bulk on level %i (%lf)\n",
              level, lbpar.gamma_bulk[level]);
      errexit();
    }
  }
  return 0;
}

#endif // LB_ADAPTIVE
