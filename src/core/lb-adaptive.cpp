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
#include "communication.hpp"
#include "constraint.hpp"
#include "lb-adaptive-gpu.hpp"
#include "lb-adaptive.hpp"
#include "lb-boundaries.hpp"
#include "lb-d3q19.hpp"
#include "lb.hpp"
#include "p4est_dd.hpp"
#include "p4est_utils.hpp"
#include "random.hpp"
#include "thermostat.hpp"
#include "utils.hpp"

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>

#ifdef LB_ADAPTIVE

/* Code duplication from lb.cpp */
/* For the D3Q19 model most functions have a separate implementation
 * where the coefficients and the velocity vectors are hardcoded
 * explicitly. This saves a lot of multiplications with 1's and 0's
 * thus making the code more efficient. */
#ifndef D3Q19
#define D3Q19
#endif // D3Q19

#if (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) && !defined(GAUSSRANDOM))
#define FLATNOISE
#endif // (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) &&
       // !defined(GAUSSRANDOM))

/* "external variables" */
p8est_connectivity_t *conn;
p8est_t *lb_p8est = 0;
p8est_ghost_t *lbadapt_ghost;
p8est_ghostvirt_t *lbadapt_ghost_virt;
p8est_mesh_t *lbadapt_mesh;
std::vector<std::vector<lbadapt_payload_t>> lbadapt_local_data;
std::vector<std::vector<lbadapt_payload_t>> lbadapt_ghost_data;
int lb_conn_brick[3] = {0, 0, 0};
double coords_for_regional_refinement[6] = {
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max()};

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

  p4est_utils_allocate_levelwise_storage(lbadapt_local_data, lbadapt_mesh,
                                         true);

  int coarsest_level_ghost = -1;
  /** ghost */
  for (int level = 0; level < P8EST_MAXLEVEL; ++level) {
    if ((((lbadapt_mesh->ghost_level + level)->elem_count > 0) ||
         ((lbadapt_mesh->virtual_glevels + level)->elem_count > 0)) &&
        coarsest_level_ghost == -1) {
      coarsest_level_ghost = level;
    }
  }
  if (coarsest_level_ghost == -1) {
    return;
  } else {
    p4est_utils_allocate_levelwise_storage(lbadapt_ghost_data, lbadapt_mesh,
                                           false);
  }

#ifdef LB_ADAPTIVE_GPU
  local_num_quadrants = lbadapt_mesh->local_num_quadrants;
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
  for (int i = 0; i < lbmodel.n_veloc; i++) {
    data->lbfluid[0][i] = 0.;
    data->lbfluid[1][i] = 0.;
  }

#ifndef LB_ADAPTIVE_GPU
  // ints
  data->lbfields.recalc_fields = 1;
  data->lbfields.has_force = 0;

  // 1D "array"
  data->lbfields.rho[0] = 0.;

  // 3D arrays
  for (int i = 0; i < 3; i++) {
    data->lbfields.j[i] = 0;
    data->lbfields.force[i] = 0;
#ifdef IMMERSED_BOUNDARY
    data->lbfields.force_buf[i] = 0;
#endif // IMMERSED_BOUNDARY
  }

  // 6D array
  for (int i = 0; i < 6; i++) {
    data->lbfields.pi[i] = 0;
  }
#endif // LB_ADAPTIVE_GPU
}

#ifndef LB_ADAPTIVE_GPU
void lbadapt_set_force(lbadapt_payload_t *data, int level)
#else // LB_ADAPTIVE_GPU
void lbadapt_set_force(lbadapt_patch_cell_t *data, int level)
#endif // LB_ADAPTIVE_GPU
{
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

#ifdef EXTERNAL_FORCES
// unit conversion: force density
#ifdef LB_ADAPTIVE_GPU
  data->force[0] =
      prefactors[level] * lbpar.ext_force[0] * SQR(h_max) * SQR(lbpar.tau);
  data->force[1] =
      prefactors[level] * lbpar.ext_force[1] * SQR(h_max) * SQR(lbpar.tau);
  data->force[2] =
      prefactors[level] * lbpar.ext_force[2] * SQR(h_max) * SQR(lbpar.tau);
#else  // LB_ADAPTIVE_GPU
  data->lbfields.force[0] =
      prefactors[level] * lbpar.ext_force[0] * SQR(h_max) * SQR(lbpar.tau);
  data->lbfields.force[1] =
      prefactors[level] * lbpar.ext_force[1] * SQR(h_max) * SQR(lbpar.tau);
  data->lbfields.force[2] =
      prefactors[level] * lbpar.ext_force[2] * SQR(h_max) * SQR(lbpar.tau);
#endif // LB_ADAPTIVE_GPU
#else  // EXTERNAL_FORCES
#ifdef LB_ADAPTIVE_GPU
  data->force[0] = 0.0;
  data->force[1] = 0.0;
  data->force[2] = 0.0;
#else  // LB_ADAPTIVE_GPU
  data->lbfields.force[0] = 0.0;
  data->lbfields.force[1] = 0.0;
  data->lbfields.force[2] = 0.0;
  data->lbfields.has_force = 0;
#endif // LB_ADAPTIVE_GPU
#endif // EXTERNAL_FORCES
}

void lbadapt_init() {
  if (lbadapt_local_data.empty()) {
    lbadapt_allocate_data();
  }
  int status;
  lbadapt_payload_t *data;
  for (int level = 0; level < P8EST_MAXLEVEL; ++level) {
    status = 0;
    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        lb_p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
        P8EST_TRAVERSE_LOCALGHOST, P8EST_TRAVERSE_REALVIRTUAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        if (!mesh_iter->current_is_ghost) {
          data =
              &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
                  mesh_iter)];
        } else {
          data =
              &lbadapt_ghost_data[level][p8est_meshiter_get_current_storage_id(
                  mesh_iter)];
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
    p8est_meshiter_destroy(mesh_iter);
  }
} // lbadapt_init();

void lbadapt_reinit_parameters() {
  for (int i = lbpar.max_refinement_level; 0 <= i; --i) {
    prefactors[i] = 1 << (lbpar.max_refinement_level - i);

#ifdef LB_ADAPTIVE_GPU
    double h = (double)P8EST_QUADRANT_LEN(i) /
               ((double)LBADAPT_PATCHSIZE * (double)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
    double h = (double)P8EST_QUADRANT_LEN(i) / (double)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU
    if (lbpar.viscosity[0] > 0.0) {
      gamma_shear[i] =
          1. -
          2. / (6. * lbpar.viscosity[0] * prefactors[i] * lbpar.tau / (SQR(h)) +
                1.);
      if (abs(gamma_shear[i] >= 1.0)) {
        fprintf(stderr, "Error when setting relaxation parameter gamma_shear "
                        "for level %i (%lf)\n",
                i, gamma_shear[i]);
        errexit();
      }
    }
    if (lbpar.bulk_viscosity[0] > 0.0) {
      gamma_bulk[i] = 1. -
                      2. / (9. * lbpar.bulk_viscosity[0] * lbpar.tau /
                                (prefactors[i] * SQR(h)) +
                            1.);
      if (abs(gamma_shear[i] >= 1.0)) {
        fprintf(stderr, "Error when setting relaxation parameter gamma_bulk "
                        "for level %i (%lf)\n",
                i, gamma_bulk[i]);
        errexit();
      }
    }
  }
#ifdef LB_ADAPTIVE_GPU
  memcpy(lbpar.prefactors, prefactors, P8EST_MAXLEVEL * sizeof(lb_float));
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

  for (int level = 0; level < P8EST_MAXLEVEL; ++level) {
    status = 0;

    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        lb_p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
        P8EST_TRAVERSE_LOCALGHOST, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        if (!mesh_iter->current_is_ghost) {
          data =
              &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
                  mesh_iter)];
        } else {
          data =
              &lbadapt_ghost_data[level][p8est_meshiter_get_current_storage_id(
                  mesh_iter)];
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
    p8est_meshiter_destroy(mesh_iter);
  }
}

void lbadapt_reinit_fluid_per_cell() {
  if (lbadapt_local_data.empty()) {
    lbadapt_allocate_data();
  }
  int status;
  lbadapt_payload_t *data;
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU
  for (int level = 0; level < P8EST_MAXLEVEL; ++level) {
    status = 0;
#ifdef LB_ADAPTIVE_GPU
    lb_float h = (lb_float)P8EST_QUADRANT_LEN(level) /
                 ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
    lb_float h = (lb_float)P8EST_QUADRANT_LEN(level) / (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        lb_p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
        P8EST_TRAVERSE_LOCALGHOST, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        if (!mesh_iter->current_is_ghost) {
          data =
              &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
                  mesh_iter)];
        } else {
          data =
              &lbadapt_ghost_data[level][p8est_meshiter_get_current_storage_id(
                  mesh_iter)];
        }
        // convert rho to lattice units
        lb_float rho = lbpar.rho[0] * h_max * h_max * h_max;
        // start with fluid at rest and no stress
        lb_float j[3] = {0., 0., 0.};
        lb_float pi[6] = {0., 0., 0., 0., 0., 0.};
#ifndef LB_ADAPTIVE_GPU
        lbadapt_calc_n_from_rho_j_pi(data->lbfluid, rho, j, pi, h);
#ifdef LB_BOUNDARIES
        data->lbfields.boundary = 0;
#endif // LB_BOUNDARIES
#else  // LB_ADAPTIVE_GPU
        for (int patch_z = 1; patch_z <= LBADAPT_PATCHSIZE; ++patch_z) {
          for (int patch_y = 1; patch_y <= LBADAPT_PATCHSIZE; ++patch_y) {
            for (int patch_x = 1; patch_x <= LBADAPT_PATCHSIZE; ++patch_x) {
              lbadapt_calc_n_from_rho_j_pi(
                  data->patch[patch_x][patch_y][patch_z].lbfluid, rho, j, pi,
                  h);
              data->patch[patch_x][patch_y][patch_z].boundary = 0;
            }
          }
        }
#endif // LB_ADAPTIVE_GPU
      }
    }
    p8est_meshiter_destroy(mesh_iter);
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
  double new_mp[3];
  int new_boundary;
  p4est_utils_get_midpoint(p4est_new, which_tree, quad_new, new_mp);
  new_boundary = lbadapt_is_boundary(new_mp);

  if (new_boundary) {
    // if cell is a boundary cell initialize data to 0.
    init_to_zero(data_new);
    data_new->lbfields.boundary = new_boundary;
  } else {
    // else calculate arithmetic mean of population of smaller quadrants
    for (int vel = 0; vel < lbmodel.n_veloc; ++vel) {
      data_new->lbfluid[0][vel] += 0.125 * data_old->lbfluid[0][vel];
    }
  }

  // finally re-initialize local force
  // TODO: Can this be optimized? E.g. by has_force flag?
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
  lb_float new_mp[3];
  int new_boundary;
  p4est_utils_get_midpoint(p4est_new, which_tree, quad_new, new_mp);
  new_boundary = lbadapt_is_boundary(new_mp);

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
  double dist, dist_tmp, dist_vec[3];
  dist = DBL_MAX;
  int the_boundary = -1;

  for (int n = 0; n < n_lb_boundaries; ++n) {
    switch (lb_boundaries[n].type) {
    case LB_BOUNDARY_WAL:
      calculate_wall_dist((Particle *)NULL, pos, (Particle *)NULL,
                          &lb_boundaries[n].c.wal, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_SPH:
      calculate_sphere_dist((Particle *)NULL, pos, (Particle *)NULL,
                            &lb_boundaries[n].c.sph, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_CYL:
      calculate_cylinder_dist((Particle *)NULL, pos, (Particle *)NULL,
                              &lb_boundaries[n].c.cyl, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_RHOMBOID:
      calculate_rhomboid_dist((Particle *)NULL, pos, (Particle *)NULL,
                              &lb_boundaries[n].c.rhomboid, &dist_tmp,
                              dist_vec);
      break;

    case LB_BOUNDARY_POR:
      calculate_pore_dist((Particle *)NULL, pos, (Particle *)NULL,
                          &lb_boundaries[n].c.pore, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_STOMATOCYTE:
      calculate_stomatocyte_dist((Particle *)NULL, pos, (Particle *)NULL,
                                 &lb_boundaries[n].c.stomatocyte, &dist_tmp,
                                 dist_vec);
      break;

    case LB_BOUNDARY_HOLLOW_CONE:
      calculate_hollow_cone_dist((Particle *)NULL, pos, (Particle *)NULL,
                                 &lb_boundaries[n].c.hollow_cone, &dist_tmp,
                                 dist_vec);
      break;

    default:
      runtimeErrorMsg() << "lbboundary type " << lb_boundaries[n].type
                        << " not implemented in lb_init_boundaries()\n";
    }

    if (dist_tmp < dist) {
      dist = dist_tmp;
      the_boundary = n;
    }
  }

  if (dist <= 0 && n_lb_boundaries > 0) {
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
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      lb_p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL, P8EST_TRAVERSE_PARBOUNDINNER);

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
          // TODO: Implement that

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
  p8est_meshiter_destroy(mesh_iter);
}
#endif // LB_ADAPTIVE_GPU

/*** Load Balance ***/
int lbadapt_partition_weight(p8est_t *p8est, p4est_topidx_t which_tree,
                             p8est_quadrant_t *q) {
  return (
      prefactors[lbpar.base_level + (lbpar.max_refinement_level - q->level)]);
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
  lb_float midpoint[3];
  p4est_utils_get_midpoint(p8est, which_tree, q, midpoint);
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
  lb_float midpoint[3];
  int coarsen = 1;
  for (int i = 0; i < P8EST_CHILDREN; ++i) {
    p4est_utils_get_midpoint(p8est, which_tree, quads[i], midpoint);
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
  int base = P8EST_QUADRANT_LEN(q->level);
  int root = P8EST_ROOT_LEN;
  // 0.6 instead of 0.5 for stability reasons
  lb_float half_length = 0.6 * sqrt(3) * ((lb_float)base / (lb_float)root);

  lb_float midpoint[3];
  p4est_utils_get_midpoint(p8est, which_tree, q, midpoint);

  double mp[3];
  mp[0] = midpoint[0];
  mp[1] = midpoint[1];
  mp[2] = midpoint[2];

  double dist, dist_tmp, dist_vec[3];
  dist = DBL_MAX;
  std::vector<int>::iterator it;

  for (int n = 0; n < n_lb_boundaries; ++n) {
    if (exclude_in_geom_ref) {
      it = std::find(exclude_in_geom_ref->begin(), exclude_in_geom_ref->end(),
                     n);
      if (it != exclude_in_geom_ref->end()) {
        continue;
      }
    }

    switch (lb_boundaries[n].type) {
    case LB_BOUNDARY_WAL:
      calculate_wall_dist((Particle *)NULL, mp, (Particle *)NULL,
                          &lb_boundaries[n].c.wal, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_SPH:
      calculate_sphere_dist((Particle *)NULL, mp, (Particle *)NULL,
                            &lb_boundaries[n].c.sph, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_CYL:
      calculate_cylinder_dist((Particle *)NULL, mp, (Particle *)NULL,
                              &lb_boundaries[n].c.cyl, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_RHOMBOID:
      calculate_rhomboid_dist((Particle *)NULL, mp, (Particle *)NULL,
                              &lb_boundaries[n].c.rhomboid, &dist_tmp,
                              dist_vec);
      break;

    case LB_BOUNDARY_POR:
      calculate_pore_dist((Particle *)NULL, mp, (Particle *)NULL,
                          &lb_boundaries[n].c.pore, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_STOMATOCYTE:
      calculate_stomatocyte_dist((Particle *)NULL, mp, (Particle *)NULL,
                                 &lb_boundaries[n].c.stomatocyte, &dist_tmp,
                                 dist_vec);
      break;

    case LB_BOUNDARY_HOLLOW_CONE:
      calculate_hollow_cone_dist((Particle *)NULL, mp, (Particle *)NULL,
                                 &lb_boundaries[n].c.hollow_cone, &dist_tmp,
                                 dist_vec);
      break;

    default:
      runtimeErrorMsg() << "lbboundary type " << lb_boundaries[n].type
                        << " not implemented in lb_init_boundaries()\n";
    }

    if (dist_tmp < dist) {
      dist = dist_tmp;
    }
  }

  if ((std::abs(dist) <= half_length) && n_lb_boundaries > 0) {
    return 1;
  } else {
    return 0;
  }
}

int refine_inv_geometric(p8est_t *p8est, p4est_topidx_t which_tree,
                         p8est_quadrant_t *q) {
  int base = P8EST_QUADRANT_LEN(q->level);
  int root = P8EST_ROOT_LEN;
  // 0.6 instead of 0.5 for stability reasons
  double half_length = 0.6 * sqrt(3) * ((double)base / (double)root);

  lb_float midpoint[3];
  p4est_utils_get_midpoint(p8est, which_tree, q, midpoint);

  double mp[3];
  mp[0] = midpoint[0];
  mp[1] = midpoint[1];
  mp[2] = midpoint[2];

  double dist, dist_tmp, dist_vec[3];
  dist = DBL_MAX;
  std::vector<int>::iterator it;

  for (int n = 0; n < n_lb_boundaries; ++n) {
    if (exclude_in_geom_ref) {
      it = std::find(exclude_in_geom_ref->begin(), exclude_in_geom_ref->end(),
                     n);
      if (it != exclude_in_geom_ref->end()) {
        continue;
      }
    }

    switch (lb_boundaries[n].type) {
    case LB_BOUNDARY_WAL:
      calculate_wall_dist((Particle *)NULL, mp, (Particle *)NULL,
                          &lb_boundaries[n].c.wal, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_SPH:
      calculate_sphere_dist((Particle *)NULL, mp, (Particle *)NULL,
                            &lb_boundaries[n].c.sph, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_CYL:
      calculate_cylinder_dist((Particle *)NULL, mp, (Particle *)NULL,
                              &lb_boundaries[n].c.cyl, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_RHOMBOID:
      calculate_rhomboid_dist((Particle *)NULL, mp, (Particle *)NULL,
                              &lb_boundaries[n].c.rhomboid, &dist_tmp,
                              dist_vec);
      break;

    case LB_BOUNDARY_POR:
      calculate_pore_dist((Particle *)NULL, mp, (Particle *)NULL,
                          &lb_boundaries[n].c.pore, &dist_tmp, dist_vec);
      break;

    case LB_BOUNDARY_STOMATOCYTE:
      calculate_stomatocyte_dist((Particle *)NULL, mp, (Particle *)NULL,
                                 &lb_boundaries[n].c.stomatocyte, &dist_tmp,
                                 dist_vec);
      break;

    case LB_BOUNDARY_HOLLOW_CONE:
      calculate_hollow_cone_dist((Particle *)NULL, mp, (Particle *)NULL,
                                 &lb_boundaries[n].c.hollow_cone, &dist_tmp,
                                 dist_vec);
      break;

    default:
      runtimeErrorMsg() << "lbboundary type " << lb_boundaries[n].type
                        << " not implemented in lb_init_boundaries()\n";
    }

    if (dist_tmp < dist) {
      dist = dist_tmp;
    }
  }

  if ((std::abs(dist) <= half_length) && n_lb_boundaries > 0) {
    return 0;
  } else {
    return 1;
  }
}

/*** HELPER FUNCTIONS ***/
int lbadapt_calc_n_from_rho_j_pi(lb_float datafield[2][19], lb_float rho,
                                 lb_float *j, lb_float *pi, lb_float h) {
  int i;
  lb_float local_rho, local_j[3], local_pi[6], trace;
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  const lb_float avg_rho = lbpar.rho[0] * h_max * h_max * h_max;

  local_rho = rho;

  for (i = 0; i < 3; ++i) {
    local_j[i] = j[i];
  }

  for (i = 0; i < 6; ++i) {
    local_pi[i] = pi[i];
  }

  trace = local_pi[0] + local_pi[2] + local_pi[5];

#ifdef D3Q19
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
#else  // D3Q19
  int i;
  lb_float tmp = 0.0;
  lb_float(*c)[3] = lbmodel.c;
  lb_float(*coeff)[4] = lbmodel.coeff;

  for (i = 0; i < lbmodel.n_veloc; i++) {
    tmp = local_pi[0] * SQR(c[i][0]) +
          (2.0 * local_pi[1] * c[i][0] + local_pi[2] * c[i][1]) * c[i][1] +
          (2.0 * (local_pi[3] * c[i][0] + local_pi[4] * c[i][1]) +
           local_pi[5] * c[i][2]) *
              c[i][2];

    datafield[0][i] = coeff[i][0] * (local_rho - avg_rho);
    datafield[0][i] += coeff[i][1] * scalar(local_j, c[i]);
    datafield[0][i] += coeff[i][2] * tmp;
    datafield[0][i] += coeff[i][3] * trace;
  }
#endif // D3Q19

  return 0;
}

int lbadapt_calc_local_fields(lb_float populations[2][19], lb_float force[3],
                              int boundary, int has_force, lb_float h,
                              lb_float *rho, lb_float *j, lb_float *pi) {
  int level = log2((lb_float)(P8EST_ROOT_LEN >> P8EST_MAXLEVEL) / h);
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU
#ifdef LB_BOUNDARIES
  if (boundary) {
    // set all to 0 on boundary
    *rho = lbpar.rho[0] * h_max * h_max * h_max;
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

  *rho = cpmode[0] + lbpar.rho[0] * h_max * h_max * h_max;

  j[0] = cpmode[1];
  j[1] = cpmode[2];
  j[2] = cpmode[3];

#ifndef EXTERNAL_FORCES
  if (has_force)
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
  modes_from_pi_eq[1] = (SQR(j[0]) - SQR(j[1])) / *rho;
  modes_from_pi_eq[2] = (scalar(j, j) - 3.0 * SQR(j[2])) / *rho;
  modes_from_pi_eq[3] = j[0] * j[1] / *rho;
  modes_from_pi_eq[4] = j[0] * j[2] / *rho;
  modes_from_pi_eq[5] = j[1] * j[2] / *rho;

  /* Now we must predict the outcome of the next collision */
  /* We immediately average pre- and post-collision. */
  cpmode[4] =
      modes_from_pi_eq[0] +
      (0.5 + 0.5 * gamma_bulk[level]) * (cpmode[4] - modes_from_pi_eq[0]);
  cpmode[5] =
      modes_from_pi_eq[1] +
      (0.5 + 0.5 * gamma_shear[level]) * (cpmode[5] - modes_from_pi_eq[1]);
  cpmode[6] =
      modes_from_pi_eq[2] +
      (0.5 + 0.5 * gamma_shear[level]) * (cpmode[6] - modes_from_pi_eq[2]);
  cpmode[7] =
      modes_from_pi_eq[3] +
      (0.5 + 0.5 * gamma_shear[level]) * (cpmode[7] - modes_from_pi_eq[3]);
  cpmode[8] =
      modes_from_pi_eq[4] +
      (0.5 + 0.5 * gamma_shear[level]) * (cpmode[8] - modes_from_pi_eq[4]);
  cpmode[9] =
      modes_from_pi_eq[5] +
      (0.5 + 0.5 * gamma_shear[level]) * (cpmode[9] - modes_from_pi_eq[5]);

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
#ifdef D3Q19
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

#ifndef OLD_FLUCT
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
#endif // !OLD_FLUCT

#else  // D3Q19
  int i, j;
  for (i = 0; i < lbmodel.n_veloc; i++) {
    mode[i] = 0.0;
    for (j = 0; j < lbmodel.n_veloc; j++) {
      mode[i] += lbmodel.e[i][j] * lbfluid[0][i][index];
    }
  }
#endif // D3Q19

#endif // LB_ADAPTIVE_GPU

  return 0;
}

int lbadapt_relax_modes(lb_float *mode, lb_float *force, lb_float h) {
#ifndef LB_ADAPTIVE_GPU
  lb_float rho, j[3], pi_eq[6];

#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  int level = log2((lb_float)(P8EST_ROOT_LEN >> P8EST_MAXLEVEL) / h);

  /* re-construct the real density
   * remember that the populations are stored as differences to their
   * equilibrium value */
  rho = mode[0] + lbpar.rho[0] * h_max * h_max * h_max;

  j[0] = mode[1];
  j[1] = mode[2];
  j[2] = mode[3];

/* if forces are present, the momentum density is redefined to
 * include one half-step of the force action.  See the
 * Chapman-Enskog expansion in [Ladd & Verberg]. */
#ifndef EXTERNAL_FORCES
  if (lbfields[index].has_force)
#endif // !EXTERNAL_FORCES
  {
    j[0] += 0.5 * force[0];
    j[1] += 0.5 * force[1];
    j[2] += 0.5 * force[2];
  }

  /* equilibrium part of the stress modes */
  pi_eq[0] = scalar(j, j) / rho;
  pi_eq[1] = (SQR(j[0]) - SQR(j[1])) / rho;
  pi_eq[2] = (scalar(j, j) - 3.0 * SQR(j[2])) / rho;
  pi_eq[3] = j[0] * j[1] / rho;
  pi_eq[4] = j[0] * j[2] / rho;
  pi_eq[5] = j[1] * j[2] / rho;

  /* relax the stress modes */
  // clang-format off
  mode[4] = pi_eq[0] + gamma_bulk[level]  * (mode[4] - pi_eq[0]);
  mode[5] = pi_eq[1] + gamma_shear[level] * (mode[5] - pi_eq[1]);
  mode[6] = pi_eq[2] + gamma_shear[level] * (mode[6] - pi_eq[2]);
  mode[7] = pi_eq[3] + gamma_shear[level] * (mode[7] - pi_eq[3]);
  mode[8] = pi_eq[4] + gamma_shear[level] * (mode[8] - pi_eq[4]);
  mode[9] = pi_eq[5] + gamma_shear[level] * (mode[9] - pi_eq[5]);
// clang-format on

#ifndef OLD_FLUCT
  /* relax the ghost modes (project them out) */
  /* ghost modes have no equilibrium part due to orthogonality */
  // clang-format off
  mode[10] = gamma_odd  * mode[10];
  mode[11] = gamma_odd  * mode[11];
  mode[12] = gamma_odd  * mode[12];
  mode[13] = gamma_odd  * mode[13];
  mode[14] = gamma_odd  * mode[14];
  mode[15] = gamma_odd  * mode[15];
  mode[16] = gamma_even * mode[16];
  mode[17] = gamma_even * mode[17];
  mode[18] = gamma_even * mode[18];
// clang-format on
#endif // !OLD_FLUCT

#endif // LB_ADAPTIVE_GPU

  return 0;
}

int lbadapt_thermalize_modes(lb_float *mode) {
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  lb_float fluct[6];
#ifdef GAUSSRANDOM
  lb_float rootrho_gauss =
      sqrt(fabs(mode[0] + lbpar.rho[0] * h_max * h_max * h_max));

  /* stress modes */
  mode[4] += (fluct[0] = rootrho_gauss * lb_phi[4] * gaussian_random());
  mode[5] += (fluct[1] = rootrho_gauss * lb_phi[5] * gaussian_random());
  mode[6] += (fluct[2] = rootrho_gauss * lb_phi[6] * gaussian_random());
  mode[7] += (fluct[3] = rootrho_gauss * lb_phi[7] * gaussian_random());
  mode[8] += (fluct[4] = rootrho_gauss * lb_phi[8] * gaussian_random());
  mode[9] += (fluct[5] = rootrho_gauss * lb_phi[9] * gaussian_random());

#ifndef OLD_FLUCT
  /* ghost modes */
  mode[10] += rootrho_gauss * lb_phi[10] * gaussian_random();
  mode[11] += rootrho_gauss * lb_phi[11] * gaussian_random();
  mode[12] += rootrho_gauss * lb_phi[12] * gaussian_random();
  mode[13] += rootrho_gauss * lb_phi[13] * gaussian_random();
  mode[14] += rootrho_gauss * lb_phi[14] * gaussian_random();
  mode[15] += rootrho_gauss * lb_phi[15] * gaussian_random();
  mode[16] += rootrho_gauss * lb_phi[16] * gaussian_random();
  mode[17] += rootrho_gauss * lb_phi[17] * gaussian_random();
  mode[18] += rootrho_gauss * lb_phi[18] * gaussian_random();
#endif // !OLD_FLUCT

#elif defined(GAUSSRANDOMCUT)
  lb_float rootrho_gauss =
      sqrt(fabs(mode[0] + lbpar.rho[0] * h_max * h_max * h_max));

  /* stress modes */
  mode[4] += (fluct[0] = rootrho_gauss * lb_phi[4] * gaussian_random_cut());
  mode[5] += (fluct[1] = rootrho_gauss * lb_phi[5] * gaussian_random_cut());
  mode[6] += (fluct[2] = rootrho_gauss * lb_phi[6] * gaussian_random_cut());
  mode[7] += (fluct[3] = rootrho_gauss * lb_phi[7] * gaussian_random_cut());
  mode[8] += (fluct[4] = rootrho_gauss * lb_phi[8] * gaussian_random_cut());
  mode[9] += (fluct[5] = rootrho_gauss * lb_phi[9] * gaussian_random_cut());

#ifndef OLD_FLUCT
  /* ghost modes */
  mode[10] += rootrho_gauss * lb_phi[10] * gaussian_random_cut();
  mode[11] += rootrho_gauss * lb_phi[11] * gaussian_random_cut();
  mode[12] += rootrho_gauss * lb_phi[12] * gaussian_random_cut();
  mode[13] += rootrho_gauss * lb_phi[13] * gaussian_random_cut();
  mode[14] += rootrho_gauss * lb_phi[14] * gaussian_random_cut();
  mode[15] += rootrho_gauss * lb_phi[15] * gaussian_random_cut();
  mode[16] += rootrho_gauss * lb_phi[16] * gaussian_random_cut();
  mode[17] += rootrho_gauss * lb_phi[17] * gaussian_random_cut();
  mode[18] += rootrho_gauss * lb_phi[18] * gaussian_random_cut();
#endif // OLD_FLUCT

#elif defined(FLATNOISE)
  lb_float rootrho =
      sqrt(fabs(12.0 * (mode[0] + lbpar.rho[0] * h_max * h_max * h_max)));

  /* stress modes */
  mode[4] += (fluct[0] = rootrho * lb_phi[4] * (d_random() - 0.5));
  mode[5] += (fluct[1] = rootrho * lb_phi[5] * (d_random() - 0.5));
  mode[6] += (fluct[2] = rootrho * lb_phi[6] * (d_random() - 0.5));
  mode[7] += (fluct[3] = rootrho * lb_phi[7] * (d_random() - 0.5));
  mode[8] += (fluct[4] = rootrho * lb_phi[8] * (d_random() - 0.5));
  mode[9] += (fluct[5] = rootrho * lb_phi[9] * (d_random() - 0.5));

#ifndef OLD_FLUCT
  /* ghost modes */
  mode[10] += rootrho * lb_phi[10] * (d_random() - 0.5);
  mode[11] += rootrho * lb_phi[11] * (d_random() - 0.5);
  mode[12] += rootrho * lb_phi[12] * (d_random() - 0.5);
  mode[13] += rootrho * lb_phi[13] * (d_random() - 0.5);
  mode[14] += rootrho * lb_phi[14] * (d_random() - 0.5);
  mode[15] += rootrho * lb_phi[15] * (d_random() - 0.5);
  mode[16] += rootrho * lb_phi[16] * (d_random() - 0.5);
  mode[17] += rootrho * lb_phi[17] * (d_random() - 0.5);
  mode[18] += rootrho * lb_phi[18] * (d_random() - 0.5);
#endif // !OLD_FLUCT
#else  // GAUSSRANDOM
#error No noise type defined for the CPU LB
#endif // GAUSSRANDOM

#ifdef ADDITIONAL_CHECKS
  rancounter += 15;
#endif // ADDITIONAL_CHECKS

  return 0;
}

int lbadapt_apply_forces(lb_float *mode, lb_float *f, lb_float h) {
  lb_float rho, u[3], C[6];

#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  int level = log2((lb_float)(P8EST_ROOT_LEN >> P8EST_MAXLEVEL) / h);

  rho = mode[0] + lbpar.rho[0] * h_max * h_max * h_max;

  /* hydrodynamic momentum density is redefined when external forces present
   */
  u[0] = (mode[1] + 0.5 * f[0]) / rho;
  u[1] = (mode[2] + 0.5 * f[1]) / rho;
  u[2] = (mode[3] + 0.5 * f[2]) / rho;

  C[0] = (1. + gamma_bulk[level]) * u[0] * f[0] +
         1. / 3. * (gamma_bulk[level] - gamma_shear[level]) * scalar(u, f);
  C[2] = (1. + gamma_bulk[level]) * u[1] * f[1] +
         1. / 3. * (gamma_bulk[level] - gamma_shear[level]) * scalar(u, f);
  C[5] = (1. + gamma_bulk[level]) * u[2] * f[2] +
         1. / 3. * (gamma_bulk[level] - gamma_shear[level]) * scalar(u, f);
  C[1] = 0.5 * (1. + gamma_shear[level]) * (u[0] * f[1] + u[1] * f[0]);
  C[3] = 0.5 * (1. + gamma_shear[level]) * (u[0] * f[2] + u[2] * f[0]);
  C[4] = 0.5 * (1. + gamma_shear[level]) * (u[1] * f[2] + u[2] * f[1]);

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
  f[0] = prefactors[level] * lbpar.ext_force[0] * SQR(h_max) * SQR(lbpar.tau);
  f[1] = prefactors[level] * lbpar.ext_force[1] * SQR(h_max) * SQR(lbpar.tau);
  f[2] = prefactors[level] * lbpar.ext_force[2] * SQR(h_max) * SQR(lbpar.tau);
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

#ifdef P4EST_ENABLE_DEBUG
  int neighbor_cnt = 0;
#endif // P4EST_ENABLE_DEBUG

  // copy resting population
  currCellData->lbfluid[1][0] = currCellData->lbfluid[0][0];

  // stream to surrounding cells
  for (int dir_ESPR = 1; dir_ESPR < lbmodel.n_veloc; ++dir_ESPR) {
    // set neighboring cell information in iterator
    int dir_p4est = ci_to_p4est[(dir_ESPR - 1)];
    p8est_meshiter_set_neighbor_quad_info(mesh_iter, dir_p4est);

    if (mesh_iter->neighbor_qid != -1) {
#ifdef P4EST_ENABLE_DEBUG
      ++neighbor_cnt;
#endif
      int inv_neigh_dir_p4est = mesh_iter->neighbor_entity_index;
      int inv_neigh_dir_ESPR = p4est_to_ci[inv_neigh_dir_p4est];
      assert(inv[dir_ESPR] == inv_neigh_dir_ESPR);
      assert(dir_ESPR == inv[inv_neigh_dir_ESPR]);

      if (mesh_iter->neighbor_is_ghost) {
        data = &lbadapt_ghost_data[mesh_iter->current_level]
                                  [p8est_meshiter_get_neighbor_storage_id(
                                      mesh_iter)];
        if (!data->lbfields.boundary) {
          // invert streaming step only if ghost neighbor is no boundary cell
          currCellData->lbfluid[1][inv[dir_ESPR]] =
              data->lbfluid[0][inv_neigh_dir_ESPR];
        }
      } else {
        data = &lbadapt_local_data[mesh_iter->current_level]
                                  [p8est_meshiter_get_neighbor_storage_id(
                                      mesh_iter)];
        data->lbfluid[1][inv[inv_neigh_dir_ESPR]] =
            currCellData->lbfluid[0][dir_ESPR];
      }
    }
  }
  P4EST_ASSERT((mesh_iter->current_level <
                p4est_utils_get_forest_info(forest_order::adaptive_LB)
                    .finest_level_global) ||
               (-1 < mesh_iter->current_vid) ||
               ((lbmodel.n_veloc - 1) == neighbor_cnt));
#endif // LB_ADAPTIVE_GPU
}

int lbadapt_calc_pop_from_modes(lb_float *populations, lb_float *m) {
#ifdef LB_ADAPTIVE
  /* normalization factors enter in the back transformation */
  for (int i = 0; i < lbmodel.n_veloc; ++i) {
    m[i] *= (1. / d3q19_modebase[19][i]);
  }

  /** perform back transformation and add weights */
  for (int i = 0; i < lbmodel.n_veloc; ++i) {
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

  lb_float modes[lbmodel.n_veloc];
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      lb_p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      quads_to_collide, P8EST_TRAVERSE_REAL, P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      if (mesh_iter->current_is_ghost) {
        data = &lbadapt_ghost_data[level][p8est_meshiter_get_current_storage_id(
            mesh_iter)];
        has_virtuals = mesh_iter->mesh->virtual_gflags[mesh_iter->current_qid];
      } else {
        data = &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
            mesh_iter)];
        has_virtuals = mesh_iter->mesh->virtual_qflags[mesh_iter->current_qid];
      }
#ifdef LB_BOUNDARIES
      if (!data->lbfields.boundary)
#endif // LB_BOUNDARIES
      {
        /* calculate modes locally */
        lbadapt_calc_modes(data->lbfluid, modes);

        /* deterministic collisions */
        lbadapt_relax_modes(modes, data->lbfields.force, h);

        /* fluctuating hydrodynamics */
        if (fluct) {
          lbadapt_thermalize_modes(modes);
        }

/* apply forces */
#ifdef EXTERNAL_FORCES
        lbadapt_apply_forces(modes, data->lbfields.force, h);
#else  // EXTERNAL_FORCES
        // forces from MD-Coupling
        if (data->lbfields.has_force) {
          lbadapt_apply_forces(modes, &data->lbfields.force, h);
        }
#endif // EXTERNAL_FORCES

        lbadapt_calc_pop_from_modes(data->lbfluid[0], modes);
      }
      if (has_virtuals) {
        lbadapt_populate_virtuals(mesh_iter);
      }
    }
  }
  p8est_meshiter_destroy(mesh_iter);
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
    virtual_sid = mesh_iter->mesh->quad_gvirtual_offset[parent_qid];
    source_data =
        &lbadapt_ghost_data[mesh_iter->current_level]
                           [p8est_meshiter_get_current_storage_id(mesh_iter)];
    virtual_data = &lbadapt_ghost_data[lvl][virtual_sid];
  } else {
    virtual_sid = mesh_iter->mesh->quad_qvirtual_offset[parent_qid];
    source_data =
        &lbadapt_local_data[mesh_iter->current_level]
                           [p8est_meshiter_get_current_storage_id(mesh_iter)];
    virtual_data = &lbadapt_local_data[lvl][virtual_sid];
  }
  for (int i = 0; i < P8EST_CHILDREN; ++i) {
    std::memcpy(virtual_data->lbfluid[0], source_data->lbfluid[0],
                lbmodel.n_veloc * sizeof(lb_float));
    std::memcpy(virtual_data->lbfluid[1], virtual_data->lbfluid[0],
                lbmodel.n_veloc * sizeof(lb_float));
    // make sure that boundary is not occupied by some memory clutter
    std::memcpy(virtual_data->lbfields.force, source_data->lbfields.force,
                P8EST_DIM * sizeof(double));
    virtual_data->lbfields.boundary = source_data->lbfields.boundary;
    ++virtual_data;
  }
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_stream(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  lbadapt_payload_t *data;
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      lb_p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REALVIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      data =
          &lbadapt_local_data[level]
                             [p8est_meshiter_get_current_storage_id(mesh_iter)];
      if (!data->lbfields.boundary) {
        lbadapt_pass_populations(mesh_iter, data);
      }
    }
  }
  p8est_meshiter_destroy(mesh_iter);
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_bounce_back(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  lbadapt_payload_t *data, *currCellData;

#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  // vector of inverse c_i, 0 is inverse to itself.
  // clang-format off
  const int inv[] = { 0,
                      2,  1,  4,  3,  6,  5,
                      8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17 };
  // clang-format on

  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      lb_p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REALVIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      currCellData =
          &lbadapt_local_data[level]
                             [p8est_meshiter_get_current_storage_id(mesh_iter)];

#ifdef D3Q19
#ifndef PULL
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

        if (mesh_iter->neighbor_qid != -1) {
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
            data = &lbadapt_ghost_data[level]
                                      [p8est_meshiter_get_neighbor_storage_id(
                                          mesh_iter)];
          } else {
            data = &lbadapt_local_data[level]
                                      [p8est_meshiter_get_neighbor_storage_id(
                                          mesh_iter)];
          }

          // case 1
          if (!mesh_iter->neighbor_is_ghost &&
              currCellData->lbfields.boundary) {
            if (!data->lbfields.boundary) {
              // calculate population shift (velocity boundary condition)
              population_shift = 0;
              for (int l = 0; l < 3; ++l) {
                population_shift -=
                    h_max * h_max * h_max * lbpar.rho[0] * 2 *
                    lbmodel.c[dir_ESPR][l] * lbmodel.w[dir_ESPR] *
                    lb_boundaries[currCellData->lbfields.boundary - 1]
                        .velocity[l] /
                    lbmodel.c_sound_sq;
              }

              // sum up the force that the fluid applies on the boundary
              for (int l = 0; l < 3; ++l) {
                lb_boundaries[currCellData->lbfields.boundary - 1].force[l] +=
                    (2 * currCellData->lbfluid[1][dir_ESPR] +
                     population_shift) *
                    lbmodel.c[dir_ESPR][l];
              }
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
                    h_max * h_max * h_max * lbpar.rho[0] * 2 *
                    lbmodel.c[inv[dir_ESPR]][l] * lbmodel.w[inv[dir_ESPR]] *
                    lb_boundaries[data->lbfields.boundary - 1].velocity[l] /
                    lbmodel.c_sound_sq;
              }

              currCellData->lbfluid[1][dir_ESPR] =
                  currCellData->lbfluid[0][inv[dir_ESPR]] + population_shift;
            } else {
              currCellData->lbfluid[1][dir_ESPR] = 0.0;
            }
          }
        } // if (mesh_iter->neighbor_qid) != -1
      }   // for (int dir_ESPR = 1; dir_ESPR < 19; ++dir_ESPR)
#else     // !PULL
#error Bounce back boundary conditions are only implemented for PUSH scheme!
#endif // !PULL
#else  // D3Q19
#error Bounce back boundary conditions are only implemented for D3Q19!
#endif // D3Q19
    }
  }
  p8est_meshiter_destroy(mesh_iter);
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_update_populations_from_virtuals(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  int parent_sid;
  lbadapt_payload_t *data, *parent_data;
  int vel;
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      lb_p8est, lbadapt_ghost, lbadapt_mesh, level + 1, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCALGHOST, P8EST_TRAVERSE_VIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      // virtual quads are local if their parent is local, ghost analogous
      if (!mesh_iter->current_is_ghost) {
        parent_sid = mesh_iter->mesh->quad_qreal_offset[mesh_iter->current_qid];
        data = &lbadapt_local_data[level + 1]
                                  [p8est_meshiter_get_current_storage_id(
                                      mesh_iter)];
        parent_data = &lbadapt_local_data[level][parent_sid];
      } else {
        parent_sid = mesh_iter->mesh->quad_greal_offset[mesh_iter->current_qid];
        data = &lbadapt_ghost_data[level + 1]
                                  [p8est_meshiter_get_current_storage_id(
                                      mesh_iter)];
        parent_data = &lbadapt_ghost_data[level][parent_sid];
      }
      for (vel = 0; vel < lbmodel.n_veloc; ++vel) {
        if (mesh_iter->current_vid == 0) {
          parent_data->lbfluid[1][vel] = 0.;
        }
        parent_data->lbfluid[1][vel] += 0.125 * data->lbfluid[0][vel];
      }
    }
  }

  p8est_meshiter_destroy(mesh_iter);
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_swap_pointers(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  lbadapt_payload_t *data;
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      lb_p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REALVIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      if (!mesh_iter->current_is_ghost) {
        data = &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
            mesh_iter)];
      } else {
        data = &lbadapt_ghost_data[level][p8est_meshiter_get_current_storage_id(
            mesh_iter)];
      }
      std::swap(data->lbfluid[0], data->lbfluid[1]);
    }
  }
  p8est_meshiter_destroy(mesh_iter);
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

  /* get boundary status */
  for (level = forest.coarsest_level_local; level <= forest.finest_level_local;
       ++level) {
    status = 0;
    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        lb_p8est, lbadapt_ghost, lbadapt_mesh, level, lbadapt_ghost->btype,
        P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);
    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
            mesh_iter)];
#ifndef LB_ADAPTIVE_GPU
        /* just grab the value of each cell and pass it into solution vector */
        bnd = data->lbfields.boundary;
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
    p8est_meshiter_destroy(mesh_iter);
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

  double h_max = (double)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                 ((double)LBADAPT_PATCHSIZE * (double)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (double)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (double)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  for (level = forest.coarsest_level_local; level <= forest.finest_level_local;
       ++level) {
    status = 0;
    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        lb_p8est, lbadapt_ghost, lbadapt_mesh, level, lbadapt_ghost->btype,
        P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
            mesh_iter)];

        double avg_rho = lbpar.rho[0] * h_max * h_max * h_max;

#ifndef LB_ADAPTIVE_GPU
        if (data->lbfields.boundary) {
          dens = 0;
        } else {
          // clang-format off
          dens = avg_rho
               + data->lbfluid[0][ 0] + data->lbfluid[0][ 1]
               + data->lbfluid[0][ 2] + data->lbfluid[0][ 3]
               + data->lbfluid[0][ 4] + data->lbfluid[0][ 5]
               + data->lbfluid[0][ 6] + data->lbfluid[0][ 7]
               + data->lbfluid[0][ 8] + data->lbfluid[0][ 9]
               + data->lbfluid[0][10] + data->lbfluid[0][11]
               + data->lbfluid[0][12] + data->lbfluid[0][13]
               + data->lbfluid[0][14] + data->lbfluid[0][15]
               + data->lbfluid[0][16] + data->lbfluid[0][17]
               + data->lbfluid[0][18];
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
    p8est_meshiter_destroy(mesh_iter);
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

  double h_max = (double)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                 ((double)LBADAPT_PATCHSIZE * (double)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU
  for (level = forest.coarsest_level_local; level <= forest.finest_level_local;
       ++level) {
    status = 0;
    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        lb_p8est, lbadapt_ghost, lbadapt_mesh, level, lbadapt_ghost->btype,
        P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

#ifdef LB_ADAPTIVE_GPU
    double h = (double)P8EST_QUADRANT_LEN(level) /
               ((double)LBADAPT_PATCHSIZE * (double)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
    double h = (double)P8EST_QUADRANT_LEN(level) / (double)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
            mesh_iter)];

        /* calculate values to write */
        double rho;
        double j[3];

#ifndef LB_ADAPTIVE_GPU
        lbadapt_calc_local_fields(data->lbfluid, data->lbfields.force,
                                  data->lbfields.boundary,
                                  data->lbfields.has_force, h, &rho, j, NULL);

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
                  &tmp_rho, tmp_j, NULL);

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
    p8est_meshiter_destroy(mesh_iter);
  }
}

/** Check if velocity for a given quadrant is set and if not set it
 * @param[in]       qid        Quadrant whose velocity is needed
 * @param[in]       data       Numerical payload of quadrant
 * @param[in, out]  vel        Vector of velocities that are already known
 */
void check_vel(int qid, double h, lbadapt_payload_t *data,
               std::vector<std::array<double, 3>> &vel) {
  double rho;
  if (vel[qid] == std::array<double, 3>{std::numeric_limits<double>::min(),
                                        std::numeric_limits<double>::min(),
                                        std::numeric_limits<double>::min()}) {
    lbadapt_calc_local_fields(data->lbfluid, data->lbfields.force,
                              data->lbfields.boundary, data->lbfields.has_force,
                              h, &rho, vel[qid].data(), 0);

    lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                     ((lb_float)P8EST_ROOT_LEN);
    vel[qid][0] = vel[qid][0] / rho * h_max / lbpar.tau;
    vel[qid][1] = vel[qid][1] / rho * h_max / lbpar.tau;
    vel[qid][2] = vel[qid][2] / rho * h_max / lbpar.tau;
  }
}

void lbadapt_calc_vorticity(p8est_t *p4est,
                            std::vector<std::array<double, 3>> &vort,
                            std::string filename) {
  P4EST_ASSERT(vort.size() == p4est->local_num_quadrants);
  // use dynamic programming to calculate fluid velocities
  std::vector<std::array<double, 3>> fluid_vel(
      p4est->local_num_quadrants + lbadapt_ghost->ghosts.elem_count,
      std::array<double, 3>({std::numeric_limits<double>::min(),
                             std::numeric_limits<double>::min(),
                             std::numeric_limits<double>::min()}));
  int nq, enc;
  int c_level;
  const std::array<int, 3> neighbor_dirs = std::array<int, 3>({1, 3, 5});
  sc_array_t *neighbor_qids = sc_array_new(sizeof(int));
  sc_array_t *neighbor_encs = sc_array_new(sizeof(int));
  std::array<std::vector<int>, 3> n_qids;
  std::array<double, 3> mesh_width;
  double h;
  p8est_quadrant_t *quad;
  lbadapt_payload_t *data, *currCellData;

  for (int qid = 0; qid < p4est->local_num_quadrants; ++qid) {
    // get neighboring quads and their velocity
    for (int dir_idx = 0; dir_idx < neighbor_dirs.size(); ++dir_idx) {
      sc_array_truncate(neighbor_qids);
      sc_array_truncate(neighbor_encs);

      quad = p8est_mesh_get_quadrant(lb_p8est, lbadapt_mesh, qid);
      c_level = quad->level;
      currCellData =
          &lbadapt_local_data[c_level][lbadapt_mesh->quad_qreal_offset[qid]];
      h = (double)P8EST_QUADRANT_LEN(c_level) / (double)P8EST_ROOT_LEN;
      check_vel(qid, h, currCellData, fluid_vel);

      p8est_mesh_get_neighbors(p4est, lbadapt_ghost, lbadapt_mesh, qid, -1,
                               neighbor_dirs[dir_idx], 0, 0, neighbor_encs,
                               neighbor_qids);
      P4EST_ASSERT(0 <= neighbor_qids->elem_count &&
                   neighbor_qids->elem_count <= P8EST_HALF);
      for (int i = 0; i < neighbor_qids->elem_count; ++i) {
        nq = *(int *)sc_array_index(neighbor_qids, i);
        enc = *(int *)sc_array_index(neighbor_encs, i);
        bool is_ghost = (enc < 0);
        if (!is_ghost) {
          quad = p8est_mesh_get_quadrant(lb_p8est, lbadapt_mesh, nq);
          data = &lbadapt_local_data[quad->level]
                                    [lbadapt_mesh->quad_qreal_offset[nq]];
        } else {
          quad = p4est_quadrant_array_index(&lbadapt_ghost->ghosts, nq);
          data = &lbadapt_ghost_data[quad->level]
                                    [lbadapt_mesh->quad_greal_offset[nq]];
        }
        h = (double)P8EST_QUADRANT_LEN(quad->level) / (double)P8EST_ROOT_LEN;
        if (is_ghost)
          nq += p4est->local_num_quadrants;
        check_vel(nq, h, data, fluid_vel);
        n_qids[dir_idx].push_back(nq);
      }
      mesh_width[dir_idx] =
          0.5 *
          (((double)P8EST_QUADRANT_LEN(c_level) / (double)P8EST_ROOT_LEN) +
           ((double)P8EST_QUADRANT_LEN(quad->level) / (double)P8EST_ROOT_LEN));
    }
    // calculate vorticity from velocity
    vort[qid][0] = 0;
    if (!currCellData->lbfields.boundary) {
      for (int i = 0; i < n_qids[0].size(); ++i) {
        vort[qid][0] += ((1. / n_qids[0].size()) *
                         (((fluid_vel[n_qids[0][i]][2] - fluid_vel[qid][2]) /
                           mesh_width[1]) -
                          ((fluid_vel[n_qids[0][i]][1] - fluid_vel[qid][1]) /
                           mesh_width[2])));
      }
    }
    n_qids[0].clear();
    vort[qid][1] = 0;
    if (!currCellData->lbfields.boundary) {
      for (int i = 0; i < n_qids[1].size(); ++i) {
        vort[qid][1] += ((1. / n_qids[1].size()) *
                         (((fluid_vel[n_qids[1][i]][0] - fluid_vel[qid][0]) /
                           mesh_width[2]) -
                          ((fluid_vel[n_qids[1][i]][2] - fluid_vel[qid][2]) /
                           mesh_width[0])));
      }
    }
    n_qids[1].clear();
    vort[qid][2] = 0;
    if (!currCellData->lbfields.boundary) {
      for (int i = 0; i < n_qids[2].size(); ++i) {
        vort[qid][2] += ((1. / n_qids[2].size()) *
                         (((fluid_vel[n_qids[2][i]][1] - fluid_vel[qid][1]) /
                           mesh_width[0]) -
                          ((fluid_vel[n_qids[2][i]][0] - fluid_vel[qid][0]) /
                           mesh_width[1])));
      }
    }
    n_qids[2].clear();
  }

  // vtk output
  // FIXME remove DEBUG
  if (filename == "") {
    filename = "vorticity_" + std::to_string((int)(sim_time / time_step));
  }
  sc_array_t *vorticity, *velocity, *qid;
  p4est_locidx_t num_cells = p4est->local_num_quadrants;
  vorticity = sc_array_new_size(sizeof(double), P8EST_DIM * num_cells);
  velocity = sc_array_new_size(sizeof(double), P8EST_DIM * num_cells);
  qid = sc_array_new_size(sizeof(double), num_cells);
  for (int i = 0; i < num_cells; ++i) {
    double *ins = (double *)sc_array_index(vorticity, 3 * i);
    std::memcpy(ins, vort[i].data(), 3 * sizeof(double));
    ins = (double *)sc_array_index(qid, i);
    *ins = i + p4est->global_first_quadrant[p4est->mpirank];
  }

  lbadapt_get_velocity_values(velocity);

  /* create VTK output context and set its parameters */
  p8est_vtk_context_t *context =
      p8est_vtk_context_new(lb_p8est, filename.c_str());
  p8est_vtk_context_set_scale(context, 1); /* quadrants at full scale */

  /* begin writing the output files */
  context = p8est_vtk_write_header(context);
  SC_CHECK_ABORT(context != NULL,
                 P8EST_STRING "_vtk: Error writing vtk header");
  // clang-format off
  context = p8est_vtk_write_cell_dataf(context,
                                       1, /* write tree indices */
                                       1, /* write the refinement level */
                                       1, /* write the mpi process id */
                                       0, /* do not wrap the mpi rank */
                                       1, /* no custom cell scalar data */
                                       2, /* write velocity and vorticity as
                                             vector cell data */
                                       "qid", qid,
                                       "velocity", velocity,
                                       "vorticity", vorticity, context);
  // clang-format on
  SC_CHECK_ABORT(context != NULL, P8EST_STRING "_vtk: Error writing cell data");

  const int retval = p8est_vtk_write_footer(context);
  SC_CHECK_ABORT(!retval, P8EST_STRING "_vtk: Error writing footer");

  /* free memory */
  sc_array_destroy(velocity);
  sc_array_destroy(vorticity);
  sc_array_destroy(qid);
  // FIXME remove DEBUG

  sc_array_destroy(neighbor_qids);
  sc_array_destroy(neighbor_encs);
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
    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        lb_p8est, lbadapt_ghost, lbadapt_mesh, level, lbadapt_ghost->btype,
        P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        assert(!mesh_iter->current_is_ghost);
        data = &lbadapt_local_data[level][p8est_meshiter_get_current_storage_id(
            mesh_iter)];

#ifndef LB_ADAPTIVE_GPU
        double midpoint[3];
        p4est_utils_get_midpoint(mesh_iter, midpoint);

        data->lbfields.boundary = lbadapt_is_boundary(midpoint);
#else  // LB_ADAPTIVE_GPU
        get_front_lower_left(mesh_iter, xyz_quad);
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

    p8est_meshiter_destroy(mesh_iter);

    p8est_ghostvirt_exchange_data(
        lb_p8est, lbadapt_ghost_virt, level, sizeof(lbadapt_payload_t),
        (void **)local_pointer.data(), (void **)ghost_pointer.data());
  }
}

void lbadapt_calc_local_rho(p8est_meshiter_t *mesh_iter, lb_float *rho) {
#ifndef LB_ADAPTIVE_GPU
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  lbadapt_payload_t *data;
  data = &lbadapt_local_data[mesh_iter->current_level]
                            [p8est_meshiter_get_neighbor_storage_id(mesh_iter)];

  lb_float avg_rho = lbpar.rho[0] * h_max * h_max * h_max;

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
  data = &lbadapt_local_data[mesh_iter->current_level]
                            [p8est_meshiter_get_neighbor_storage_id(mesh_iter)];

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
/** should no longer be needed */
void lbadapt_set_recalc_fields(p8est_iter_volume_info_t *info,
                               void *user_data) {
#ifndef LB_ADAPTIVE_GPU
  p8est_quadrant_t *q = info->quad;
  lbadapt_payload_t *data = (lbadapt_payload_t *)q->p.user_data;

  data->lbfields.recalc_fields = 1;
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_calc_local_rho(p8est_iter_volume_info_t *info, void *user_data) {
#ifndef LB_ADAPTIVE_GPU
  lb_float *rho = (lb_float *)user_data; /* passed lb_float to fill */
  p8est_quadrant_t *q = info->quad;      /* get current global cell id */
  lbadapt_payload_t *data =
      (lbadapt_payload_t *)q->p.user_data; /* payload of cell */
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(lbpar.max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

#ifndef D3Q19
#error Only D3Q19 is implemened!
#endif // D3Q19

  // unit conversion: mass density
  if (!(lattice_switch & LATTICE_LB)) {
    runtimeErrorMsg() << "Error in lb_calc_local_rho in " << __FILE__
                      << __LINE__ << ": CPU LB not switched on.";
    *rho = 0;
    return;
  }

  lb_float avg_rho = lbpar.rho[0] * h_max * h_max * h_max;

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

void lbadapt_calc_local_pi(p8est_iter_volume_info_t *info, void *user_data) {
#ifndef LB_ADAPTIVE_GPU
  lb_float *bnd_vals = (lb_float *)user_data; /* passed array to fill */
  p8est_quadrant_t *q = info->quad;           /* get current global cell id */
  p4est_topidx_t which_tree = info->treeid;   /* get current tree id */
  p4est_locidx_t local_id = info->quadid;     /* get cell id w.r.t. tree-id */
  p8est_tree_t *tree;
  lbadapt_payload_t *data =
      (lbadapt_payload_t *)q->p.user_data; /* payload of cell */

  lb_float bnd; /* local meshwidth */
  p4est_locidx_t arrayoffset;

  tree = p8est_tree_array_index(lb_p8est->trees, which_tree);
  local_id += tree->quadrants_offset; /* now the id is relative to the MPI
                                         process */
  arrayoffset =
      local_id; /* each local quadrant has 2^d (P8EST_CHILDREN) values in
                   u_interp */

  /* just grab the u value of each cell and pass it into solution vector
   */
  bnd = data->lbfields.boundary;
  bnd_vals[arrayoffset] = bnd;
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_dump2file(p8est_iter_volume_info_t *info, void *user_data) {
#ifndef LB_ADAPTIVE_GPU
  lbadapt_payload_t *data = (lbadapt_payload_t *)info->quad->p.user_data;
  p8est_quadrant_t *q = info->quad;

  std::string *filename = (std::string *)user_data;
  std::ofstream myfile;
  myfile.open(*filename, std::ofstream::out | std::ofstream::app);
  myfile << "id: " << info->quadid
         << "; coords: " << (q->x / (1 << (P8EST_MAXLEVEL - q->level))) << ", "
         << (q->y / (1 << (P8EST_MAXLEVEL - q->level))) << ", "
         << (q->z / (1 << (P8EST_MAXLEVEL - q->level)))
         << "; boundary: " << data->lbfields.boundary << std::endl
         << " - distributions: pre streaming: ";
  for (int i = 0; i < 19; ++i) {
    myfile << data->lbfluid[0][i] << " - ";
  }
  myfile << std::endl << "post streaming: ";
  for (int i = 0; i < 19; ++i) {
    myfile << data->lbfluid[1][i] << " - ";
  }
  myfile << std::endl << std::endl;

  myfile.flush();
  myfile.close();
#endif // LB_ADAPTIVE_GPU
}

int64_t lbadapt_get_global_idx(p8est_quadrant_t *q, p4est_topidx_t tree) {
  int x, y, z;
  double xyz[3];
  p8est_qcoord_to_vertex(conn, tree, q->x, q->y, q->z, xyz);
  x = xyz[0] * (1 << lbpar.max_refinement_level);
  y = xyz[1] * (1 << lbpar.max_refinement_level);
  z = xyz[2] * (1 << lbpar.max_refinement_level);

  return p4est_utils_cell_morton_idx(x, y, z);
}

int64_t lbadapt_get_global_idx(p8est_quadrant_t *q, p4est_topidx_t tree,
                               int displace[3]) {
  int x, y, z;
  double xyz[3];
  p8est_qcoord_to_vertex(conn, tree, q->x, q->y, q->z, xyz);
  x = xyz[0] * (1 << lbpar.max_refinement_level) + displace[0];
  y = xyz[1] * (1 << lbpar.max_refinement_level) + displace[1];
  z = xyz[2] * (1 << lbpar.max_refinement_level) + displace[2];

  int ub = lb_conn_brick[0] * (1 << lbpar.max_refinement_level);
  if (x >= ub)
    x -= ub;
  if (x < 0)
    x += ub;
  ub = lb_conn_brick[1] * (1 << lbpar.max_refinement_level);
  if (y >= ub)
    y -= ub;
  if (y < 0)
    y += ub;
  ub = lb_conn_brick[2] * (1 << lbpar.max_refinement_level);
  if (z >= ub)
    z -= ub;
  if (z < 0)
    z += ub;

  return p4est_utils_cell_morton_idx(x, y, z);
}

int64_t lbadapt_map_pos_to_ghost(double pos[3]) {
  p8est_quadrant_t *q;
  int idx_a, idx_b;
  int xid, yid, zid;
  for (int d = 0; d < 3; ++d) {
    if (pos[d] > (box_l[d] + box_l[d] * ROUND_ERROR_PREC))
      return -1;
    if (pos[d] < -box_l[d] * ROUND_ERROR_PREC)
      return -1;
  }
  xid = (pos[0]) * (1 << lbpar.max_refinement_level);
  yid = (pos[1]) * (1 << lbpar.max_refinement_level);
  zid = (pos[2]) * (1 << lbpar.max_refinement_level);
  int64_t pidx = p4est_utils_cell_morton_idx(xid, yid, zid);
  int64_t qidx, zlvlfill;
  // idea: iterate through all ghost quadrants and check if searched position
  //       is contained in current ghost quadrant. if yes, return ghost index
  for (size_t i = 0; i < lbadapt_ghost->ghosts.elem_count; ++i) {
    q = p8est_quadrant_array_index(&lbadapt_ghost->ghosts, i);
    qidx = lbadapt_get_global_idx(q, q->p.piggy3.which_tree);
    zlvlfill = 1 << (3 * (lbpar.max_refinement_level - q->level));
    if (qidx <= pidx && pidx < qidx + zlvlfill)
      idx_a = i;
  }
  idx_b =
      p4est_utils_pos_qid_ghost(forest_order::adaptive_LB, lbadapt_ghost, pos);
  P4EST_ASSERT(idx_a == idx_b);
  return idx_b;
}

int lbadapt_interpolate_pos_adapt(double pos[3], lbadapt_payload_t *nodes[20],
                                  double delta[20], int level[20]) {
  static const int nidx[8][7] = {
      {0, 2, 14, 4, 10, 6, 18}, // left, front, bottom
      {1, 2, 15, 4, 11, 6, 19}, // right, front, bottom
      {0, 3, 16, 4, 10, 7, 20}, // left, back, bottom
      {1, 3, 17, 4, 11, 7, 21}, // right, back, bottom
      {0, 2, 14, 5, 12, 8, 22}, // left, front, top
      {1, 2, 15, 5, 13, 8, 23}, // right, front, top
      {0, 3, 16, 5, 12, 9, 24}, // left, back, top
      {1, 3, 17, 5, 13, 9, 25}, // right, back, top
  };
  static const int didx[8][3] = {
      {0, 1, 2}, {3, 1, 2}, {0, 4, 2}, {3, 4, 2},
      {0, 1, 5}, {3, 1, 5}, {0, 4, 5}, {3, 4, 5},
  };
  int64_t qidx = p4est_utils_pos_quad_ext(forest_order::adaptive_LB, pos);
  if (qidx < 0) {
    int ncnt = lbadapt_interpolate_pos_ghost(pos, nodes, delta, level);
    if (ncnt > 0)
      return ncnt;
    fprintf(stderr, "Particle not in local LB domain ");
    fprintf(stderr, "%i : %li [%lf %lf %lf], ", this_node, qidx, pos[0], pos[1],
            pos[2]);
#ifdef DD_P4EST
    fprintf(stderr, "belongs to MD process %i\n", dd_p4est_pos_to_proc(pos));
#endif // DD_P4EST
    errexit();
    return -1;
  }
  int lvl, sid;
  p8est_quadrant_t *quad;
  quad = p8est_mesh_get_quadrant(lb_p8est, lbadapt_mesh, qidx);
  lvl = quad->level;
  sid = lbadapt_mesh->quad_qreal_offset[qidx];
  nodes[0] = &lbadapt_local_data[lvl][sid];
  level[0] = lvl;
  int corner = 0;
  double delta_loc[6];
  for (int d = 0; d < 3; ++d) {
    double dis = pos[d] * (double)(1 << lvl);
    dis = dis - floor(dis) + 0.5;
    if (dis > 1.0) { // right neighbor
      corner |= 1 << d;
      dis = 2.0 - dis;
    }
    delta_loc[d] = dis;
    delta_loc[d + 3] = 1.0 - dis;
  }
  delta[0] =
      delta_loc[didx[0][0]] * delta_loc[didx[0][1]] * delta_loc[didx[0][2]];
  int ncnt = 1;
  int xidx = -1;
  int yidx = -1;
  int zidx = -1;
  for (int i = 0; i < 7; ++i) {
    sc_array_t *ne, *ni;
    ne = sc_array_new(sizeof(int));
    ni = sc_array_new(sizeof(int));
    p8est_mesh_get_neighbors(lb_p8est, lbadapt_ghost, lbadapt_mesh, qidx, -1,
                             nidx[corner][i], 0, NULL, ne, ni);
    if (ne->elem_count == 0) {
      switch (i) {
      case 2: // X-Y edge
        if (xidx >= 0)
          delta[xidx] +=
              (delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
               delta_loc[didx[i + 1][2]]);
        else if (yidx >= 0)
          delta[yidx] +=
              (delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
               delta_loc[didx[i + 1][2]]);
        break;
      case 4: // X-Z edge
        if (xidx >= 0)
          delta[xidx] +=
              (delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
               delta_loc[didx[i + 1][2]]);
        else if (zidx >= 0)
          delta[zidx] +=
              (delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
               delta_loc[didx[i + 1][2]]);
        break;
      case 5: // Y-Z edge
        if (yidx >= 0)
          delta[xidx] +=
              (delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
               delta_loc[didx[i + 1][2]]);
        else if (zidx >= 0)
          delta[yidx] +=
              (delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
               delta_loc[didx[i + 1][2]]);
        break;
      case 6: // X-Y-Z corner
        if (xidx >= 0)
          delta[xidx] +=
              (delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
               delta_loc[didx[i + 1][2]]);
        else if (yidx >= 0)
          delta[xidx] +=
              (delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
               delta_loc[didx[i + 1][2]]);
        else if (zidx >= 0)
          delta[yidx] +=
              (delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
               delta_loc[didx[i + 1][2]]);
        break;
      default:
        printf("A LB cell neighbor is missing over a face\n");
        break;
      };
    }
    for (size_t n = 0; n < ne->elem_count; ++n) {
      int nidx = *((int *)sc_array_index_int(ni, n));
      int enc = *((int *)sc_array_index_int(ne, n));
      if (enc > 0) {                   // local quadrant
        if (enc >= 25 && enc <= 120) { // double size quad over face
          if (i == 0)
            xidx = ncnt;
          if (i == 1)
            yidx = ncnt;
          if (i == 3)
            zidx = ncnt;
        }
        quad = p8est_mesh_get_quadrant(lb_p8est, lbadapt_mesh, nidx);
        lvl = quad->level;
        sid = lbadapt_mesh->quad_qreal_offset[nidx];
        nodes[ncnt] = &lbadapt_local_data[lvl][sid];
        delta[ncnt] = delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
                      delta_loc[didx[i + 1][2]];
        delta[ncnt] = delta[ncnt] / (double)(ne->elem_count);
        level[ncnt] = lvl;
        ncnt += 1;
      } else {                           // ghost quadrant
        if (enc >= -120 && enc <= -25) { // double size quad over face
          if (i == 0)
            xidx = ncnt;
          if (i == 1)
            yidx = ncnt;
          if (i == 3)
            zidx = ncnt;
        }
        quad = p8est_quadrant_array_index(&lbadapt_ghost->ghosts, nidx);
        lvl = quad->level;
        sid = lbadapt_mesh->quad_greal_offset[nidx];
        nodes[ncnt] = &lbadapt_ghost_data[lvl][sid];
        delta[ncnt] = delta_loc[didx[i + 1][0]] * delta_loc[didx[i + 1][1]] *
                      delta_loc[didx[i + 1][2]];
        delta[ncnt] = delta[ncnt] / (double)(ne->elem_count);
        level[ncnt] = lvl;
        ncnt += 1;
      }
    }
    sc_array_destroy(ne);
    sc_array_destroy(ni);
  }
  double dsum = 1.0;
  for (int i = 0; i < ncnt; ++i)
    dsum -= delta[i];
  if (abs(dsum) > ROUND_ERROR_PREC)
    printf("dsum is larger than round error precision: %le\n", dsum);
  if (ncnt > 20)
    printf("too many neighbours\n");
  return ncnt;
}

int lbadapt_interpolate_pos_ghost(double opos[3], lbadapt_payload_t *nodes[20],
                                  double delta[20], int level[20]) {
  static const int didx[8][3] = {
      {0, 1, 2}, {3, 1, 2}, {0, 4, 2}, {3, 4, 2},
      {0, 1, 5}, {3, 1, 5}, {0, 4, 5}, {3, 4, 5},
  };

  int fold[3] = {0, 0, 0};
  double pos[3] = {opos[0], opos[1], opos[2]};
  fold_position(pos, fold);

  int64_t qidx = lbadapt_map_pos_to_ghost(pos);

  if (qidx < 0)
    return 0;

  int lvl, sid, tree, zarea, zsize;
  p8est_quadrant_t *quad;

  quad = p8est_quadrant_array_index(&lbadapt_ghost->ghosts, qidx);
  tree = quad->p.piggy3.which_tree;
  lvl = quad->level;
  sid = lbadapt_mesh->quad_greal_offset[qidx];
  nodes[0] = &lbadapt_ghost_data[lvl][sid];
  level[0] = lvl;

  int corner = 0;
  double delta_loc[6];
  for (int d = 0; d < 3; ++d) {
    double dis = pos[d] * (double)(1 << lvl);
    dis = dis - floor(dis) + 0.5;
    if (dis > 1.0) { // right neighbor
      corner |= 1 << d;
      dis = 2.0 - dis;
    }
    delta_loc[d] = dis;
    delta_loc[d + 3] = 1.0 - dis;
  }
  delta[0] =
      delta_loc[didx[0][0]] * delta_loc[didx[0][1]] * delta_loc[didx[0][2]];
  zsize = 1 << (lbpar.max_refinement_level - lvl);
  zarea = zsize * zsize * zsize;

  int ncnt = 1;

  // collect over all 7 direction wrt. to corner
  int cnt_dir = 1;
  for (int dir = 1; dir < 8; ++dir) {
    int displace[3] = {0, 0, 0};
    if ((dir & 1)) {
      if ((corner & 1))
        displace[0] = zsize;
      else
        displace[0] = -zsize;
    }
    if ((dir & 2)) {
      if ((corner & 2))
        displace[1] = zsize;
      else
        displace[1] = -zsize;
    }
    if ((dir & 4)) {
      if ((corner & 4))
        displace[2] = zsize;
      else
        displace[2] = -zsize;
    }
    int64_t nidx = lbadapt_get_global_idx(quad, tree, displace);
    for (int64_t i = 0; i < lbadapt_ghost->mirrors.elem_count; ++i) {
      p8est_quadrant_t *q =
          p8est_quadrant_array_index(&lbadapt_ghost->mirrors, i);
      qidx = lbadapt_get_global_idx(q, q->p.piggy3.which_tree);
      if (qidx >= nidx && qidx < nidx + zarea) {
        sid = lbadapt_mesh->quad_qreal_offset[q->p.piggy3.local_num];
        nodes[ncnt] = &lbadapt_local_data[q->level][sid];
        level[ncnt] = q->level;
        delta[ncnt] = delta_loc[didx[dir][0]] * delta_loc[didx[dir][1]] *
                      delta_loc[didx[dir][2]];
        if (q->level > lvl) {
          if (dir == 1 || dir == 2 || dir == 4)
            delta[ncnt] *= 0.25;
          if (dir == 3 || dir == 5 || dir == 6)
            delta[ncnt] *= 0.5;
        }
        ncnt += 1;
        cnt_dir |= 1 << dir;
      }
    }
    for (int64_t i = 0; i < lbadapt_ghost->ghosts.elem_count; ++i) {
      p8est_quadrant_t *q =
          p8est_quadrant_array_index(&lbadapt_ghost->ghosts, i);
      qidx = lbadapt_get_global_idx(q, q->p.piggy3.which_tree);
      if (qidx >= nidx && qidx < nidx + zarea) {
        sid = lbadapt_mesh->quad_greal_offset[i];
        nodes[ncnt] = &lbadapt_ghost_data[q->level][sid];
        level[ncnt] = q->level;
        delta[ncnt] = delta_loc[didx[dir][0]] * delta_loc[didx[dir][1]] *
                      delta_loc[didx[dir][2]];
        if (q->level > lvl) {
          if (dir == 1 || dir == 2 || dir == 4)
            delta[ncnt] *= 0.25;
          if (dir == 3 || dir == 5 || dir == 6)
            delta[ncnt] *= 0.5;
        }
        ncnt += 1;
        cnt_dir |= 1 << dir;
      }
    }
  }
  if (cnt_dir !=
      255) { // not all neighbors for all directions exist => in outside halo
    return 0;
  }
  return ncnt;
}

#endif // LB_ADAPTIVE
