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
#include "lb-adaptive.hpp"
#include "lb-boundaries.hpp"
#include "lb-d3q19.hpp"
#include "lb.hpp"
#include "random.hpp"
#include "thermostat.hpp"
#include "utils.hpp"

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
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
p8est_t *p8est;
p8est_ghost_t *lbadapt_ghost;
p8est_ghostvirt_t *lbadapt_ghost_virt;
p8est_mesh_t *lbadapt_mesh;
lbadapt_payload_t **lbadapt_local_data;
lbadapt_payload_t **lbadapt_ghost_data;
int coarsest_level_local;
int finest_level_local;
int coarsest_level_ghost;
int finest_level_ghost;
int finest_level_global;

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
  int level;
  coarsest_level_local = -1;
  finest_level_local = -1;
  coarsest_level_ghost = -1;
  finest_level_ghost = -1;

  /** local cells */
  for (level = 0; level < P8EST_MAXLEVEL; ++level) {
    if ((((lbadapt_mesh->quad_level + level)->elem_count > 0) ||
         ((lbadapt_mesh->virtual_qlevels + level)->elem_count > 0)) &&
        coarsest_level_local == -1) {
      coarsest_level_local = level;
    }
    if ((coarsest_level_local != -1) &&
        (((lbadapt_mesh->quad_level + level)->elem_count > 0) ||
         ((lbadapt_mesh->virtual_qlevels + level)->elem_count > 0))) {
      finest_level_local = level;
    }
  }

  lbadapt_local_data = P4EST_ALLOC(
      lbadapt_payload_t *, 1 + finest_level_local - coarsest_level_local);
  for (level = coarsest_level_local; level <= finest_level_local; ++level) {
    lbadapt_local_data[level - coarsest_level_local] = P4EST_ALLOC(
        lbadapt_payload_t,
        (lbadapt_mesh->quad_level + level)->elem_count +
            P8EST_CHILDREN *
                (lbadapt_mesh->virtual_qlevels + level)->elem_count);
#if 0
      std::cout << "[p4est " << p8est->mpirank << "] Allocated space for "
                << (lbadapt_mesh->quad_level + level)->elem_count
                << " real and "
                << P8EST_CHILDREN *
                       (lbadapt_mesh->virtual_qlevels + level)->elem_count
                << " virtual local quadrants of level " << level << std::endl;
#endif // 0
  }

  /** ghost */
  for (int level = 0; level < P8EST_MAXLEVEL; ++level) {
    if ((((lbadapt_mesh->ghost_level + level)->elem_count > 0) ||
         ((lbadapt_mesh->virtual_glevels + level)->elem_count > 0)) &&
        coarsest_level_ghost == -1) {
      coarsest_level_ghost = level;
    }
    if ((coarsest_level_ghost != -1) &&
        (((lbadapt_mesh->ghost_level + level)->elem_count > 0) ||
         ((lbadapt_mesh->virtual_glevels + level)->elem_count > 0))) {
      finest_level_ghost = level;
    }
  }
  if (coarsest_level_ghost == -1) {
    return;
  }
  lbadapt_ghost_data = P4EST_ALLOC(
      lbadapt_payload_t *, 1 + finest_level_ghost - coarsest_level_ghost);
  for (level = coarsest_level_ghost; level <= finest_level_ghost; ++level) {
    lbadapt_ghost_data[level - coarsest_level_ghost] = P4EST_ALLOC(
        lbadapt_payload_t,
        (lbadapt_mesh->ghost_level + level)->elem_count +
            P8EST_CHILDREN *
                (lbadapt_mesh->virtual_glevels + level)->elem_count);
#if 0
      std::cout << "[p4est " << p8est->mpirank << "] Allocated space for "
                << (lbadapt_mesh->ghost_level + level)->elem_count
                << " real and "
                << P8EST_CHILDREN *
                       (lbadapt_mesh->virtual_glevels + level)->elem_count
                << " virtual ghost quadrants of level " << level << std::endl;
#endif // 0
  }
} // lbadapt_allocate_data();

#ifndef LB_ADAPTIVE_GPU
void init_to_zero(lbadapt_payload_t *data) {
#else  // LB_ADAPTIVE_GPU
void init_to_zero(lbadapt_patch_cell_t *data) {
#endif // LB_ADAPTIVE_GPU
  for (int i = 0; i < lbmodel.n_veloc; i++) {
    data->lbfluid[0][i] = 0.;
    data->lbfluid[1][i] = 0.;
    data->modes[i] = 0.;
  }

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
}

#ifndef LB_ADAPTIVE_GPU
void lbadapt_set_force(lbadapt_payload_t *data, int level)
#else // LB_ADAPTIVE_GPU
void lbadapt_set_force(lbadapt_patch_cell_t *data, int level)
#endif // LB_ADAPTIVE_GPU
{
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

#ifdef EXTERNAL_FORCES
  // unit conversion: force density
  data->lbfields.force[0] =
      prefactors[level] * lbpar.ext_force[0] * SQR(h_max) * SQR(lbpar.tau);
  data->lbfields.force[1] =
      prefactors[level] * lbpar.ext_force[1] * SQR(h_max) * SQR(lbpar.tau);
  data->lbfields.force[2] =
      prefactors[level] * lbpar.ext_force[2] * SQR(h_max) * SQR(lbpar.tau);
#else  // EXTERNAL_FORCES
  data->lbfields.force[0] = 0.0;
  data->lbfields.force[1] = 0.0;
  data->lbfields.force[2] = 0.0;
  data->lbfields.has_force = 0;
#endif // EXTERNAL_FORCES
}

void lbadapt_init() {
  if (lbadapt_local_data == NULL) {
    lbadapt_allocate_data();
  }
  int status;
  lbadapt_payload_t *data;
  int lvl;
  for (int level = 0; level < P8EST_MAXLEVEL; ++level) {
    status = 0;
    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
        P8EST_TRAVERSE_LOCALGHOST, P8EST_TRAVERSE_REALVIRTUAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        if (!mesh_iter->current_is_ghost) {
          lvl = level - coarsest_level_local;
          data = &lbadapt_local_data[lvl][p8est_meshiter_get_current_storage_id(
              mesh_iter)];
        } else {
          lvl = level - coarsest_level_ghost;
          data = &lbadapt_ghost_data[lvl][p8est_meshiter_get_current_storage_id(
              mesh_iter)];
        }
        data->boundary = 0;
#ifndef LB_ADAPTIVE_GPU
        init_to_zero(data);
#else  // LB_ADAPTIVE_GPU
        for (int patch_z = 0; patch_z < LBADAPT_PATCHSIZE_HALO; ++patch_z) {
          for (int patch_y = 0; patch_y < LBADAPT_PATCHSIZE_HALO; ++patch_y) {
            for (int patch_x = 0; patch_x < LBADAPT_PATCHSIZE_HALO; ++patch_x) {
              init_to_zero(&data->patch[patch_x][patch_y][patch_z]);
            }
          }
        }
#endif // LB_ADAPTIVE_GPU
      }
    }
    p8est_meshiter_destroy(mesh_iter);
  }
} // lbadapt_init();

void lbadapt_reinit_force_per_cell() {
  if (lbadapt_local_data == NULL) {
    lbadapt_allocate_data();
  }
  int status;
  int lvl;
  lbadapt_payload_t *data;

  for (int level = 0; level < P8EST_MAXLEVEL; ++level) {
    status = 0;

    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
        P8EST_TRAVERSE_LOCALGHOST, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        if (!mesh_iter->current_is_ghost) {
          lvl = level - coarsest_level_local;
          data = &lbadapt_local_data[lvl][p8est_meshiter_get_current_storage_id(
              mesh_iter)];
        } else {
          lvl = level - coarsest_level_ghost;
          data = &lbadapt_ghost_data[lvl][p8est_meshiter_get_current_storage_id(
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
  if (lbadapt_local_data == NULL) {
    lbadapt_allocate_data();
  }
  int status;
  int lvl;
  lbadapt_payload_t *data;
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
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
        p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
        P8EST_TRAVERSE_LOCALGHOST, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        if (!mesh_iter->current_is_ghost) {
          lvl = level - coarsest_level_local;
          data = &lbadapt_local_data[lvl][p8est_meshiter_get_current_storage_id(
              mesh_iter)];
        } else {
          lvl = level - coarsest_level_ghost;
          data = &lbadapt_ghost_data[lvl][p8est_meshiter_get_current_storage_id(
              mesh_iter)];
        }
        // convert rho to lattice units
        lb_float rho = lbpar.rho[0] * h_max * h_max * h_max;
        // start with fluid at rest and no stress
        lb_float j[3] = {0., 0., 0.};
        lb_float pi[6] = {0., 0., 0., 0., 0., 0.};
#ifndef LB_ADAPTIVE_GPU
        lbadapt_calc_n_from_rho_j_pi(data->lbfluid, rho, j, pi, h);
#else  // LB_ADAPTIVE_GPU
        for (int patch_z = 1; patch_z <= LBADAPT_PATCHSIZE; ++patch_z) {
          for (int patch_y = 1; patch_y <= LBADAPT_PATCHSIZE; ++patch_y) {
            for (int patch_x = 1; patch_x <= LBADAPT_PATCHSIZE; ++patch_x) {
              lbadapt_calc_n_from_rho_j_pi(
                  data->patch[patch_x][patch_y][patch_z].lbfluid, rho, j, pi,
                  h);
            }
          }
        }
#endif // LB_ADAPTIVE_GPU

#ifdef LB_BOUNDARIES
        data->boundary = 0;
#endif // LB_BOUNDARIES
      }
    }
    p8est_meshiter_destroy(mesh_iter);
  }
}

int lbadapt_get_global_maxlevel() {
  int i;
  int local_res = -1;
  int global_res;
  p8est_tree_t *tree;

  /* get local max level */
  for (i = p8est->first_local_tree; i <= p8est->last_local_tree; ++i) {
    tree = p8est_tree_array_index(p8est->trees, i);
    if (local_res < tree->maxlevel) {
      local_res = tree->maxlevel;
    }
  }

  /* synchronize and return obtained result */
  sc_MPI_Allreduce(&local_res, &global_res, 1, sc_MPI_INT, sc_MPI_MAX,
                   p8est->mpicomm);

  return global_res;
}

#ifdef LB_ADAPTIVE_GPU
void lbadapt_patches_populate_halos(int level) {
  // clang-format off
  const int inv[] = {0,
                     2,  1,  4,  3,  6,  5,
                     8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17};
  // clang-format on
  lbadapt_payload_t *data, *neighbor_data;
  int status = 0;
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL, P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      if (!mesh_iter->current_is_ghost) {
        data = &lbadapt_local_data[level - coarsest_level_local]
                                  [p8est_meshiter_get_current_storage_id(
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
                &lbadapt_ghost_data[level - coarsest_level_ghost]
                                   [p8est_meshiter_get_neighbor_storage_id(
                                       mesh_iter)];
          } else {
            neighbor_data =
                &lbadapt_local_data[level - coarsest_level_local]
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
            // iterate over the other to indices, keeping the original direction
            // constant.
            iter_max_x = iter_max_y = iter_max_z = LBADAPT_PATCHSIZE;
            r_offset_x = r_offset_y = r_offset_z = 1;
            w_offset_x = w_offset_y = w_offset_z = 0;
            if (4 == (dir_p4est & 4)) {
              iter_max_z = 1;
              r_offset_z = (dir_p4est % 2 == 0 ? LBADAPT_PATCHSIZE : 1);
              w_offset_z = (dir_p4est % 2 == 0 ? 0 : LBADAPT_PATCHSIZE + 1);
            } else {
              if (2 == (dir_p4est & 2)) {
                iter_max_y = 1;
                r_offset_y = (dir_p4est % 2 == 0 ? LBADAPT_PATCHSIZE : 1);
                w_offset_y = (dir_p4est % 2 == 0 ? 0 : LBADAPT_PATCHSIZE + 1);
              } else {
                iter_max_x = 1;
                r_offset_x = (dir_p4est % 2 == 0 ? LBADAPT_PATCHSIZE : 1);
                w_offset_x = (dir_p4est % 2 == 0 ? 0 : LBADAPT_PATCHSIZE + 1);
              }
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
  return (prefactors[lbpar.base_level + (max_refinement_level - q->level)]);
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
  lbadapt_get_midpoint(p8est, which_tree, q, midpoint);
  if (midpoint[2] < 0.5) {
    return 1;
  }
  return 0;
}

int refine_geometric(p8est_t *p8est, p4est_topidx_t which_tree,
                     p8est_quadrant_t *q) {
  int base = P8EST_QUADRANT_LEN(q->level);
  int root = P8EST_ROOT_LEN;
  // 0.6 instead of 0.5 for stability reasons
  lb_float half_length = 0.6 * sqrt(3) * ((lb_float)base / (lb_float)root);

  lb_float midpoint[3];
  lbadapt_get_midpoint(p8est, which_tree, q, midpoint);

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

/*** HELPER FUNCTIONS ***/
void lbadapt_get_midpoint(p8est_t *p8est, p4est_topidx_t which_tree,
                          p8est_quadrant_t *q, lb_float xyz[3]) {
  int base = P8EST_QUADRANT_LEN(q->level);
  int root = P8EST_ROOT_LEN;
  lb_float half_length = ((lb_float)base / (lb_float)root) * 0.5;
  double tmp[3];

  p8est_qcoord_to_vertex(p8est->connectivity, which_tree, q->x, q->y, q->z,
                         tmp);
  for (int i = 0; i < P8EST_DIM; ++i) {
    xyz[i] = tmp[i] + half_length;
  }
}

void lbadapt_get_midpoint(p8est_meshiter_t *mesh_iter, lb_float *xyz) {
  int base = P8EST_QUADRANT_LEN(mesh_iter->current_level);
  int root = P8EST_ROOT_LEN;
  lb_float half_length = ((lb_float)base / (lb_float)root) * 0.5;

  p8est_quadrant_t *q = p8est_mesh_get_quadrant(
      mesh_iter->p4est, mesh_iter->mesh, mesh_iter->current_qid);
  double tmp[3];
  p8est_qcoord_to_vertex(p8est->connectivity,
                         mesh_iter->mesh->quad_to_tree[mesh_iter->current_qid],
                         q->x, q->y, q->z, tmp);

  for (int i = 0; i < P8EST_DIM; ++i) {
    xyz[i] = tmp[i] + half_length;
  }
}

void lbadapt_get_front_lower_left(p8est_meshiter_t *mesh_iter, lb_float *xyz) {
  p8est_quadrant_t *q = p8est_mesh_get_quadrant(
      mesh_iter->p4est, mesh_iter->mesh, mesh_iter->current_qid);
  double tmp[3];
  p8est_qcoord_to_vertex(p8est->connectivity,
                         mesh_iter->mesh->quad_to_tree[mesh_iter->current_qid],
                         q->x, q->y, q->z, tmp);
  for (int i = 0; i < P8EST_DIM; ++i) {
    xyz[i] = tmp[i];
  }
}

void lbadapt_get_front_lower_left(p8est_t *p8est, p4est_topidx_t which_tree,
                                  p8est_quadrant_t *q, double *xyz) {
  p8est_qcoord_to_vertex(p8est->connectivity, which_tree, q->x, q->y, q->z,
                         xyz);
}

int lbadapt_calc_n_from_rho_j_pi(lb_float datafield[2][19], lb_float rho,
                                 lb_float *j, lb_float *pi, lb_float h) {
  int i;
  lb_float local_rho, local_j[3], local_pi[6], trace;
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
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

int lbadapt_calc_local_fields(lb_float mode[19], lb_float force[3],
                              int boundary, int has_force, lb_float h,
                              lb_float *rho, lb_float *j, lb_float *pi) {
  int level = log2((lb_float)(P8EST_ROOT_LEN >> P8EST_MAXLEVEL) / h);
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
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
  for (int i = 0; i < 19; ++i) {
    cpmode[i] = mode[i];
  }
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
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
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
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
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

int lbadapt_apply_forces(lb_float *mode, LB_FluidNode *lbfields, lb_float h) {
  lb_float rho, u[3], C[6], *f;

#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  int level = log2((lb_float)(P8EST_ROOT_LEN >> P8EST_MAXLEVEL) / h);

  f = lbfields->force;

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
  lbfields->force[0] =
      prefactors[level] * lbpar.ext_force[0] * SQR(h_max) * SQR(lbpar.tau);
  lbfields->force[1] =
      prefactors[level] * lbpar.ext_force[1] * SQR(h_max) * SQR(lbpar.tau);
  lbfields->force[2] =
      prefactors[level] * lbpar.ext_force[2] * SQR(h_max) * SQR(lbpar.tau);
#else  // EXTERNAL_FORCES
  lbfields->force[0] = 0.0;
  lbfields->force[1] = 0.0;
  lbfields->force[2] = 0.0;
  lbfields->has_force = 0;
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

int lbadapt_calc_n_from_modes_push(p8est_meshiter_t *mesh_iter) {
#ifndef LB_ADAPTIVE_GPU
#ifdef D3Q19
  /* index of inverse velocity vector, 0 is inverse to itself. */
  // clang-format off
  int inv[] = {0,
               2,  1,  4,  3,  6,  5,
               8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17};
  // clang-format on

  /**************************************************/
  /* process current cell (backtransformation only) */
  /**************************************************/
  /* containers for accessing data of current cell through p4est_mesh */
  lbadapt_payload_t *data, *currCellData;

  currCellData =
      &lbadapt_local_data[mesh_iter->current_level - coarsest_level_local]
                         [p8est_meshiter_get_current_storage_id(mesh_iter)];
  lb_float *m = currCellData->modes;
  lb_float ghost_m[19];

  /* normalization factors enter in the back transformation */
  for (int i = 0; i < lbmodel.n_veloc; i++) {
    m[i] = (1. / d3q19_modebase[19][i]) * m[i];
  }

#ifndef OLD_FLUCT
  /* cell itself cannot be ghost cell */
  currCellData->lbfluid[1][0] = lbadapt_backTransformation(m, 0) * lbmodel.w[0];

  /*******************************/
  /* stream to surrounding cells */
  /*******************************/
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

      // if the neighboring cell is a ghost cell:
      // do not stream information into the ghost cell but perform the inverse
      // streaming step and stream information into the current cell
      // but do all this only if the neighboring cell is allowed to stream,
      // i.e. is not a boundary cell.
      if (mesh_iter->neighbor_is_ghost) {
        data =
            &lbadapt_ghost_data[mesh_iter->current_level - coarsest_level_ghost]
                               [p8est_meshiter_get_neighbor_storage_id(
                                   mesh_iter)];
        if (!data->boundary) {
          /* obtain and normalize ghost modes */
          if (mesh_iter->neighbor_vid == -1) {
            // if interacting with a real quadrant: do inverse streaming
            // operation
            for (int i = 0; i < lbmodel.n_veloc; ++i) {
              ghost_m[i] = data->modes[i];
              ghost_m[i] = (1. / d3q19_modebase[19][i]) * ghost_m[i];
            }
            currCellData->lbfluid[1][inv[dir_ESPR]] =
                lbadapt_backTransformation(ghost_m, inv_neigh_dir_ESPR) *
                lbmodel.w[inv_neigh_dir_ESPR];
          } else {
            // else pass inverse population
            currCellData->lbfluid[1][inv[dir_ESPR]] =
                data->lbfluid[0][inv_neigh_dir_ESPR];
          }
        }
      } else {
        data =
            &lbadapt_local_data[mesh_iter->current_level - coarsest_level_local]
                               [p8est_meshiter_get_neighbor_storage_id(
                                   mesh_iter)];
        data->lbfluid[1][inv[inv_neigh_dir_ESPR]] =
            lbadapt_backTransformation(m, dir_ESPR) * lbmodel.w[dir_ESPR];
      }
    }
  }
#else //! OLD_FLUCT
#error OLD_FLUCT not implemented
#endif // !OLD_FLUCT
#endif // D3Q19
#endif // LB_ADAPTIVE_GPU

  return 0;
}

void lbadapt_pass_populations(p8est_meshiter_t *mesh_iter) {
#ifndef LB_ADAPTIVE_GPU
  // clang-format off
  int inv[] = {0,
               2,  1,  4,  3,  6,  5,
               8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17};
  // clang-format on
  lbadapt_payload_t *data, *currCellData;
  currCellData =
      &lbadapt_local_data[mesh_iter->current_level - coarsest_level_local]
                         [p8est_meshiter_get_current_storage_id(mesh_iter)];
  lb_float ghost_m[19];
  for (int dir_ESPR = 1; dir_ESPR < 19; ++dir_ESPR) {
    // set neighboring cell information in iterator
    int dir_p4est = ci_to_p4est[(dir_ESPR - 1)];
    p8est_meshiter_set_neighbor_quad_info(mesh_iter, dir_p4est);

    if (mesh_iter->neighbor_qid != -1) {
      int inv_neigh_dir_p4est = mesh_iter->neighbor_entity_index;
      int inv_neigh_dir_ESPR = p4est_to_ci[inv_neigh_dir_p4est];
      assert(inv[dir_ESPR] == inv_neigh_dir_ESPR);
      assert(dir_ESPR == inv[inv_neigh_dir_ESPR]);

      if (mesh_iter->neighbor_is_ghost) {
        // do the magic of inverting the streaming step only if the neighboring
        // cell is not a boundary cell, i.e. it is allowed to stream.
        data =
            &lbadapt_ghost_data[mesh_iter->current_level - coarsest_level_ghost]
                               [p8est_meshiter_get_neighbor_storage_id(
                                   mesh_iter)];
        if (!data->boundary) {
          if (mesh_iter->neighbor_vid == -1) {
            // neighbor is a real quadrant: do inverse streaming operation
            for (int i = 0; i < lbmodel.n_veloc; ++i) {
              ghost_m[i] = data->modes[i];
              ghost_m[i] = (1. / d3q19_modebase[19][i]) * ghost_m[i];
            }
            currCellData->lbfluid[1][inv[dir_ESPR]] =
                lbadapt_backTransformation(ghost_m, inv_neigh_dir_ESPR) *
                lbmodel.w[inv_neigh_dir_ESPR];
          } else {
            // neighbor is a virtual quadrant: pass population in the opposite
            // direction to current quadrant
            currCellData->lbfluid[1][inv[dir_ESPR]] =
                data->lbfluid[0][inv_neigh_dir_ESPR];
          }
        }
      } else {
        data =
            &lbadapt_local_data[mesh_iter->current_level - coarsest_level_local]
                               [p8est_meshiter_get_neighbor_storage_id(
                                   mesh_iter)];
        data->lbfluid[1][inv[inv_neigh_dir_ESPR]] =
            currCellData->lbfluid[0][dir_ESPR];
      }
    }
  }
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_collide(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
#ifdef LB_ADAPTIVE_GPU
  lb_float h = (lb_float)P8EST_QUADRANT_LEN(level) /
               ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h = (lb_float)P8EST_QUADRANT_LEN(level) / (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  lbadapt_payload_t *data;
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL, P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      if (mesh_iter->current_is_ghost) {
        data = &lbadapt_ghost_data[level - coarsest_level_ghost]
                                  [p8est_meshiter_get_current_storage_id(
                                      mesh_iter)];
      } else {
        data = &lbadapt_local_data[level - coarsest_level_local]
                                  [p8est_meshiter_get_current_storage_id(
                                      mesh_iter)];
      }
#ifdef LB_BOUNDARIES
      if (!data->boundary)
#endif // LB_BOUNDARIES
      {
        /* calculate modes locally */
        lbadapt_calc_modes(data->lbfluid, data->modes);

        /* deterministic collisions */
        lbadapt_relax_modes(data->modes, data->lbfields.force, h);

        /* fluctuating hydrodynamics */
        if (fluct)
          lbadapt_thermalize_modes(data->modes);

/* apply forces */
#ifdef EXTERNAL_FORCES
        lbadapt_apply_forces(data->modes, &data->lbfields, h);
#else  // EXTERNAL_FORCES
        // forces from MD-Coupling
        if (data->lbfields.has_force)
          lbadapt_apply_forces(data->modes, &data->lbfields, h);
#endif // EXTERNAL_FORCES
      }
    }
  }
  p8est_meshiter_destroy(mesh_iter);
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_populate_virtuals(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  int current_sid;
  int parent_sid;
  lbadapt_payload_t *current_data, *parent_data;
  int lvl;
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      p8est, lbadapt_ghost, lbadapt_mesh, level + 1, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_VIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    // virtual quads are local if their parent is local, ghost analogous
    if (status != P8EST_MESHITER_DONE) {
      if (!mesh_iter->current_is_ghost) {
        lvl = level - coarsest_level_local;

        parent_sid = mesh_iter->mesh->quad_qreal_offset[mesh_iter->current_qid];
        current_sid = p8est_meshiter_get_current_storage_id(mesh_iter);

        parent_data = &lbadapt_local_data[lvl][parent_sid];
        current_data = &lbadapt_local_data[lvl + 1][current_sid];
      } else {
        lvl = level - coarsest_level_ghost;

        parent_sid = mesh_iter->mesh->quad_greal_offset[mesh_iter->current_qid];
        current_sid = p8est_meshiter_get_current_storage_id(mesh_iter);

        parent_data = &lbadapt_ghost_data[lvl][parent_sid];
        current_data = &lbadapt_ghost_data[lvl + 1][current_sid];
      }
      // copy payload from coarse cell
      memcpy(current_data, parent_data, sizeof(lbadapt_payload_t));

      // calculate post_collision populations from cell
      for (int i = 0; i < lbmodel.n_veloc; ++i) {
        current_data->modes[i] *= (1. / d3q19_modebase[19][i]);
      }

      for (int i = 0; i < lbmodel.n_veloc; ++i) {
        current_data->lbfluid[0][i] =
            lbadapt_backTransformation(current_data->modes, i) * lbmodel.w[i];
      }

      // synchronize pre- and post-collision values
      memcpy(current_data->lbfluid[1], current_data->lbfluid[0],
             lbmodel.n_veloc * sizeof(lb_float));
    }
  }
  p8est_meshiter_destroy(mesh_iter);
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_stream(int level) {
#ifndef LB_ADAPTIVE_GPU
  int status = 0;
  lbadapt_payload_t *data;
  int lvl = level - coarsest_level_local;
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REALVIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      data =
          &lbadapt_local_data[lvl]
                             [p8est_meshiter_get_current_storage_id(mesh_iter)];
      if (!data->boundary) {
        if (mesh_iter->current_vid == -1) {
          lbadapt_calc_n_from_modes_push(mesh_iter);
        } else {
          lbadapt_pass_populations(mesh_iter);
        }
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
#if 0
#ifdef LB_ADAPTIVE_GPU
    lb_float h = (lb_float)P8EST_QUADRANT_LEN(level) /
               ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
    lb_float h = (lb_float)P8EST_QUADRANT_LEN(level) / (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU
#endif // 0

#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  // vector of inverse c_i, 0 is inverse to itself.
  // clang-format off
  const int inv[] = {0,
                     2,  1,  4,  3,  6,  5,
                     8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17};
  // clang-format on

  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REALVIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      currCellData =
          &lbadapt_local_data[level - coarsest_level_local]
                             [p8est_meshiter_get_current_storage_id(mesh_iter)];

#ifdef D3Q19
#ifndef PULL
      lb_float population_shift;
      lb_float local_post_collision_populations[19] = {
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
          -1, -1, -1, -1, -1, -1, -1, -1, -1};
      if (currCellData->boundary) {
        currCellData->lbfluid[1][0] = 0.0;
      }

      // We cannot copy this with minimal invasiveness, because boundary cells
      // can end in the ghost layer and p4est_iterate does not visit ghost
      // cells. Therefore we have to inspect in each cell, if it has neighbors
      // that are part of the ghost layer.
      // Then there are 2 cases where we have to bounce back:
      // 1. Current cell itself is boundary cell
      //    In this case we perform the same algorithm as in the regular grid
      // 2. Current cell is fluid cell and has ghost neighbors that are
      //    boundary cells.
      //    In this case we we have to do something different:
      //    The neighboring cell was not allowed to stream, because it is a
      //    boundary cell and in addition, we did not stream to it.
      //    So instead we have to perform the back-transformation for the
      //    populations of this cell and write our population in search
      //    direction to our inverse position.
      for (int dir_ESPR = 1; dir_ESPR < 19; ++dir_ESPR) {
        // set neighboring cell information in iterator
        int dir_p4est = ci_to_p4est[(dir_ESPR - 1)];
        p8est_meshiter_set_neighbor_quad_info(mesh_iter, dir_p4est);

        if (mesh_iter->neighbor_qid != -1) {
          int inv_neigh_dir_p4est = mesh_iter->neighbor_entity_index;
          int inv_neigh_dir_ESPR = p4est_to_ci[inv_neigh_dir_p4est];
          assert(inv[dir_ESPR] == inv_neigh_dir_ESPR);
          assert(dir_ESPR == inv[inv_neigh_dir_ESPR]);

          // get neighboring cells
          p8est_meshiter_set_neighbor_quad_info(mesh_iter, dir_p4est);
          if (mesh_iter->neighbor_qid != -1) {
            if (mesh_iter->neighbor_is_ghost) {
              data = &lbadapt_ghost_data[level - coarsest_level_ghost]
                                        [p8est_meshiter_get_neighbor_storage_id(
                                            mesh_iter)];
            } else {
              data = &lbadapt_local_data[level - coarsest_level_local]
                                        [p8est_meshiter_get_neighbor_storage_id(
                                            mesh_iter)];
            }

            // case 1
            if (!mesh_iter->neighbor_is_ghost && currCellData->boundary) {
              if (!data->boundary) {
                // calculate population shift (moving boundary)
                population_shift = 0;
                for (int l = 0; l < 3; ++l) {
                  population_shift -=
                      h_max * h_max * h_max * h_max * h_max * lbpar.rho[0] * 2 *
                      lbmodel.c[dir_ESPR][l] * lbmodel.w[dir_ESPR] *
                      lb_boundaries[currCellData->boundary - 1].velocity[l] /
                      lbmodel.c_sound_sq;
                }

                for (int l = 0; l < 3; ++l) {
                  lb_boundaries[currCellData->boundary - 1].force[l] +=
                      (2 * currCellData->lbfluid[1][dir_ESPR] +
                       population_shift) *
                      lbmodel.c[dir_ESPR][l];
                }
                data->lbfluid[1][inv[inv_neigh_dir_ESPR]] =
                    currCellData->lbfluid[1][inv[dir_ESPR]] + population_shift;
              } else {
                data->lbfluid[1][inv[inv_neigh_dir_ESPR]] =
                    currCellData->lbfluid[1][inv[dir_ESPR]] = 0.0;
              }
            }

            // case 2
            else if (mesh_iter->neighbor_is_ghost == 1 && data->boundary) {
              if (!currCellData->boundary) {
                if (-1. == local_post_collision_populations[0]) {
                  for (int i = 0; i < lbmodel.n_veloc; ++i) {
                    local_post_collision_populations[i] =
                        lbadapt_backTransformation(currCellData->modes, i) *
                        lbmodel.w[i];
                  }
                }
                // calculate population shift (moving boundary)
                population_shift = 0.;
                for (int l = 0; l < 3; l++) {
                  population_shift -=
                      h_max * h_max * h_max * h_max * h_max * lbpar.rho[0] * 2 *
                      lbmodel.c[inv[dir_ESPR]][l] * lbmodel.w[inv[dir_ESPR]] *
                      lb_boundaries[data->boundary - 1].velocity[l] /
                      lbmodel.c_sound_sq;
                }

                for (int l = 0; l < 3; ++l) {
                  lb_boundaries[data->boundary - 1].force[l] +=
                      (2 * data->lbfluid[1][inv[dir_ESPR]] + population_shift) *
                      lbmodel.c[inv[dir_ESPR]][l];
                }
                currCellData->lbfluid[1][inv[dir_ESPR]] =
                    local_post_collision_populations[dir_ESPR] +
                    population_shift;
              } else {
                currCellData->lbfluid[1][inv[dir_ESPR]] = 0.0;
              }
            }
          }
        }
      }
#else // !PULL
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
  int lvl;
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      p8est, lbadapt_ghost, lbadapt_mesh, level + 1, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_VIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      // virtual quads are local if their parent is local, ghost analogous
      if (!mesh_iter->current_is_ghost) {
        parent_sid = mesh_iter->mesh->quad_qreal_offset[mesh_iter->current_qid];
        lvl = level - coarsest_level_local;
        data =
            &lbadapt_local_data[lvl + 1][p8est_meshiter_get_current_storage_id(
                mesh_iter)];
        parent_data = &lbadapt_local_data[lvl][parent_sid];
      } else {
        parent_sid = mesh_iter->mesh->quad_greal_offset[mesh_iter->current_qid];
        lvl = level - coarsest_level_ghost;
        data =
            &lbadapt_ghost_data[lvl + 1][p8est_meshiter_get_current_storage_id(
                mesh_iter)];
        parent_data = &lbadapt_ghost_data[lvl][parent_sid];
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
  int lvl = level - coarsest_level_local;
  p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
      p8est, lbadapt_ghost, lbadapt_mesh, level, P8EST_CONNECT_EDGE,
      P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REALVIRTUAL,
      P8EST_TRAVERSE_PARBOUNDINNER);

  while (status != P8EST_MESHITER_DONE) {
    status = p8est_meshiter_next(mesh_iter);
    if (status != P8EST_MESHITER_DONE) {
      data =
          &lbadapt_local_data[lvl]
                             [p8est_meshiter_get_current_storage_id(mesh_iter)];
      std::swap(data->lbfluid[0], data->lbfluid[1]);
    }
  }
  p8est_meshiter_destroy(mesh_iter);
#endif // LB_ADAPTIVE_GPU
}

void lbadapt_get_boundary_values(sc_array_t *boundary_values) {
  int status;
  int level;
  double bnd, *bnd_ptr;
  lbadapt_payload_t *data;
#ifdef LB_ADAPTIVE_GPU
  int cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
#endif // LB_ADAPTIVE_GPU

  /* get boundary status */
  for (level = coarsest_level_local; level <= finest_level_local; ++level) {
    status = 0;
    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        p8est, lbadapt_ghost, lbadapt_mesh, level, lbadapt_ghost->btype,
        P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);
    int lvl = level - coarsest_level_local;
    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[lvl][p8est_meshiter_get_current_storage_id(
            mesh_iter)];
#ifndef LB_ADAPTIVE_GPU
        /* just grab the value of each cell and pass it into solution vector */
        bnd = data->boundary;
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
                  data->patch[patch_x][patch_y][patch_z].lbfields.boundary;
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
  int status;
  int level;
  double dens, *dens_ptr, h;
  lbadapt_payload_t *data;

#ifdef LB_ADAPTIVE_GPU
  int cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;

  double h_max = (double)P8EST_QUADRANT_LEN(max_refinement_level) /
                 ((double)LBADAPT_PATCHSIZE * (double)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max =
      (double)P8EST_QUADRANT_LEN(max_refinement_level) / (double)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  for (level = coarsest_level_local; level <= finest_level_local; ++level) {
    status = 0;
    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        p8est, lbadapt_ghost, lbadapt_mesh, level, lbadapt_ghost->btype,
        P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    int lvl = level - coarsest_level_local;
#ifdef LB_ADAPTIVE_GPU
    double h = (double)P8EST_QUADRANT_LEN(level) /
               ((double)LBADAPT_PATCHSIZE * (double)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
    double h = (double)P8EST_QUADRANT_LEN(level) / (double)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[lvl][p8est_meshiter_get_current_storage_id(
            mesh_iter)];

        double avg_rho = lbpar.rho[0] * h_max * h_max * h_max;

#ifndef LB_ADAPTIVE_GPU
        if (data->boundary) {
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
              if (data->patch[patch_x][patch_y][patch_z].lbfields.boundary) {
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
  int status;
  int level;
  double *veloc_ptr, h;
  lbadapt_payload_t *data;

#ifdef LB_ADAPTIVE_GPU
  int cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;

  double h_max = (double)P8EST_QUADRANT_LEN(max_refinement_level) /
                 ((double)LBADAPT_PATCHSIZE * (double)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU
  for (level = coarsest_level_local; level <= finest_level_local; ++level) {
    status = 0;
    p8est_meshiter_t *mesh_iter = p8est_meshiter_new_ext(
        p8est, lbadapt_ghost, lbadapt_mesh, level, lbadapt_ghost->btype,
        P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    int lvl = level - coarsest_level_local;
#ifdef LB_ADAPTIVE_GPU
    double h = (double)P8EST_QUADRANT_LEN(level) /
               ((double)LBADAPT_PATCHSIZE * (double)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
    double h = (double)P8EST_QUADRANT_LEN(level) / (double)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[lvl][p8est_meshiter_get_current_storage_id(
            mesh_iter)];

        /* calculate values to write */
        double rho;
        double j[3];

#ifndef LB_ADAPTIVE_GPU
        lbadapt_calc_local_fields(data->modes, data->lbfields.force,
                                  data->boundary, data->lbfields.has_force, h,
                                  &rho, j, NULL);

        j[0] = j[0] / rho * h_max / lbpar.tau;
        j[1] = j[1] / rho * h_max / lbpar.tau;
        j[2] = j[2] / rho * h_max / lbpar.tau;

        veloc_ptr = (lb_float *)sc_array_index(
            velocity_values, P8EST_DIM * mesh_iter->current_qid);

        /* pass it into solution vector */
        std::memcpy(veloc_ptr, j, P8EST_DIM * sizeof(lb_float));
#else  // LB_ADAPTIVE_GPU
        int patch_count = 0;
        for (int patch_z = 1; patch_z <= LBADAPT_PATCHSIZE; ++patch_z) {
          for (int patch_y = 1; patch_y <= LBADAPT_PATCHSIZE; ++patch_y) {
            for (int patch_x = 1; patch_x <= LBADAPT_PATCHSIZE; ++patch_x) {
              lb_float tmp_rho, tmp_j[3];
              lbadapt_calc_local_fields(
                  data->patch[patch_x][patch_y][patch_z].modes,
                  data->patch[patch_x][patch_y][patch_z].lbfields.force,
                  data->patch[patch_x][patch_y][patch_z].lbfields.boundary,
                  data->patch[patch_x][patch_y][patch_z].lbfields.has_force, h,
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

void lbadapt_get_boundary_status() {
  int status;
  int level;
  lbadapt_payload_t *data;

  /* set boundary status */
  for (level = coarsest_level_local; level <= finest_level_local; ++level) {
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
        p8est, lbadapt_ghost, lbadapt_mesh, level, lbadapt_ghost->btype,
        P8EST_TRAVERSE_LOCAL, P8EST_TRAVERSE_REAL,
        P8EST_TRAVERSE_PARBOUNDINNER);

    int lvl = level - coarsest_level_local;

    while (status != P8EST_MESHITER_DONE) {
      status = p8est_meshiter_next(mesh_iter);
      if (status != P8EST_MESHITER_DONE) {
        data = &lbadapt_local_data[lvl][p8est_meshiter_get_current_storage_id(
            mesh_iter)];

#ifndef LB_ADAPTIVE_GPU
        double midpoint[3];
        lbadapt_get_midpoint(mesh_iter, midpoint);

        data->boundary = lbadapt_is_boundary(midpoint);
#else  // LB_ADAPTIVE_GPU
        lbadapt_get_front_lower_left(mesh_iter, xyz_quad);
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
              data->patch[1 + patch_x][1 + patch_y][1 + patch_z]
                  .lbfields.boundary = lbadapt_is_boundary(xyz_patch);
              all_boundary = all_boundary &&
                             data->patch[1 + patch_x][1 + patch_y][1 + patch_z]
                                 .lbfields.boundary;
            }
          }
        }
        data->boundary = all_boundary;
#endif // LB_ADAPTIVE_GPU
      }
    }
    p8est_meshiter_destroy(mesh_iter);
  }
}

void lbadapt_calc_local_rho(p8est_meshiter_t *mesh_iter, lb_float *rho) {
#ifndef LB_ADAPTIVE_GPU
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU

  lbadapt_payload_t *data;
  data = &lbadapt_local_data[mesh_iter->current_level - coarsest_level_local]
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
  data = &lbadapt_local_data[mesh_iter->current_level - coarsest_level_local]
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
  lb_float h = (lb_float)P8EST_QUADRANT_LEN(q->level) /
               ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h =
      (lb_float)P8EST_QUADRANT_LEN(q->level) / (lb_float)P8EST_ROOT_LEN;
#endif // LB_ADAPTIVE_GPU
#ifdef LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
                   ((lb_float)LBADAPT_PATCHSIZE * (lb_float)P8EST_ROOT_LEN);
#else  // LB_ADAPTIVE_GPU
  lb_float h_max = (lb_float)P8EST_QUADRANT_LEN(max_refinement_level) /
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

  tree = p8est_tree_array_index(p8est->trees, which_tree);
  local_id += tree->quadrants_offset; /* now the id is relative to the MPI
                                         process */
  arrayoffset =
      local_id; /* each local quadrant has 2^d (P8EST_CHILDREN) values in
                   u_interp */

  /* just grab the u value of each cell and pass it into solution vector
   */
  bnd = data->boundary;
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
         << "; boundary: " << data->boundary << std::endl
         << " - distributions: pre streaming: ";
  for (int i = 0; i < 19; ++i) {
    myfile << data->lbfluid[0][i] << " - ";
  }
  myfile << std::endl << "post streaming: ";
  for (int i = 0; i < 19; ++i) {
    myfile << data->lbfluid[1][i] << " - ";
  }
  myfile << std::endl << "modes: ";
  for (int i = 0; i < 19; ++i) {
    myfile << data->modes[i] << " - ";
  }
  myfile << std::endl << std::endl;

  myfile.flush();
  myfile.close();
#endif // LB_ADAPTIVE_GPU
}

#ifdef LB_ADAPTIVE_GPU
lbadapt_vtk_context_t *lbadapt_vtk_context_new(const char *filename) {
  lbadapt_vtk_context_t *cont;

  assert(filename != NULL);

  /* Allocate, initialize the vtk context.  Important to zero all fields. */
  cont = P4EST_ALLOC_ZERO(lbadapt_vtk_context_t, 1);

  cont->filename = P4EST_STRDUP(filename);

#if 0
  if (sizeof(float) == sizeof(lb_float)) {
    strcpy(cont->vtk_float_name, "Float32");
  } else if (sizeof(double) == sizeof(lb_float)) {
    strcpy(cont->vtk_float_name, "Float64");
  } else {
    SC_ABORT_NOT_REACHED();
  }
#else  // 0
  strcpy(cont->vtk_float_name, "Float64");
#endif // 0

  return cont;
}

void lbadapt_vtk_context_destroy(lbadapt_vtk_context_t *context) {
  P4EST_ASSERT(context != NULL);

  /* since this function is called inside write_header and write_footer,
   * we cannot assume a consistent state of all member variables */

  P4EST_ASSERT(context->filename != NULL);
  P4EST_FREE(context->filename);

  /* deallocate node storage */
  if (context->nodes != NULL) {
    p8est_nodes_destroy(context->nodes);
  }
  P4EST_FREE(context->node_to_corner);

  /* Close all file pointers. */
  if (context->vtufile != NULL) {
    if (fclose(context->vtufile)) {
      P4EST_LERRORF(P8EST_STRING "_vtk: Error closing <%s>.\n",
                    context->vtufilename);
    }
    context->vtufile = NULL;
  }

  /* Close paraview master file */
  if (context->pvtufile != NULL) {
    /* Only the root process opens/closes these files. */
    P4EST_ASSERT(p8est->mpirank == 0);
    if (fclose(context->pvtufile)) {
      P4EST_LERRORF(P8EST_STRING "_vtk: Error closing <%s>.\n",
                    context->pvtufilename);
    }
    context->pvtufile = NULL;
  }

  /* Close visit master file */
  if (context->visitfile != NULL) {
    /* Only the root process opens/closes these files. */
    P4EST_ASSERT(p8est->mpirank == 0);
    if (fclose(context->visitfile)) {
      P4EST_LERRORF(P8EST_STRING "_vtk: Error closing <%s>.\n",
                    context->visitfilename);
    }
    context->visitfile = NULL;
  }

  /* Free context structure. */
  P4EST_FREE(context);
}

lbadapt_vtk_context_t *lbadapt_vtk_write_header(lbadapt_vtk_context_t *cont) {
  const lb_float intsize = 1.0 / P8EST_ROOT_LEN;
  int mpirank;
  const char *filename;
  const lb_float *v;
  const p4est_topidx_t *tree_to_vertex;
  p4est_topidx_t first_local_tree, last_local_tree;
  const p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  p4est_locidx_t Ncells, Ncorners;
  p8est_t *p4est;
  p8est_connectivity_t *connectivity;
  int retval;
  uint8_t *uint8_data;
  p4est_locidx_t *locidx_data;
  int xi, yi, j, k;
  int zi;
  double xyz[3]; /* 3 not P4EST_DIM */
  size_t num_quads, zz;
  p4est_topidx_t jt;
  p4est_topidx_t vt[P8EST_CHILDREN];
  p4est_locidx_t patch_count, quad_count, Npoints;
  p4est_locidx_t il, *ntc;
  double *float_data;
  sc_array_t *quadrants, *indeps;
  sc_array_t *trees;
  p8est_tree_t *tree;
  p8est_quadrant_t *quad;
  p8est_nodes_t *nodes;

  /* check a whole bunch of assertions, here and below */
  P4EST_ASSERT(cont != NULL);
  P4EST_ASSERT(!cont->writing);

  /* avoid uninitialized warning */
  for (k = 0; k < P8EST_CHILDREN; ++k) {
    vt[k] = -(k + 1);
  }

  /* from now on this context is officially in use for writing */
  cont->writing = 1;

  /* grab context variables */
  p4est = p8est;
  filename = cont->filename;
  P4EST_ASSERT(filename != NULL);

  /* grab details from the forest */
  P4EST_ASSERT(p4est != NULL);
  mpirank = p4est->mpirank;
  connectivity = p4est->connectivity;
  P4EST_ASSERT(connectivity != NULL);
  v = (lb_float *)connectivity->vertices;
  tree_to_vertex = connectivity->tree_to_vertex;

  SC_CHECK_ABORT(connectivity->num_vertices > 0,
                 "Must provide connectivity with vertex information");
  P4EST_ASSERT(v != NULL && tree_to_vertex != NULL);

  trees = p4est->trees;
  first_local_tree = p4est->first_local_tree;
  last_local_tree = p4est->last_local_tree;
  Ncells = cells_per_patch * p4est->local_num_quadrants;

  cont->num_corners = Ncorners = P8EST_CHILDREN * Ncells;
  cont->nodes = nodes = NULL;
  cont->num_points = Npoints = Ncorners;
  cont->node_to_corner = ntc = NULL;
  indeps = NULL;

  /* Have each proc write to its own file */
  snprintf(cont->vtufilename, BUFSIZ, "%s_%04d.vtu", filename, mpirank);

  /* Use "w" for writing the initial part of the file.
   * For further parts, use "r+" and fseek so write_compressed succeeds.
   */
  cont->vtufile = fopen(cont->vtufilename, "wb");
  if (cont->vtufile == NULL) {
    P4EST_LERRORF("Could not open %s for output\n", cont->vtufilename);
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }

  fprintf(cont->vtufile, "<?xml version=\"1.0\"?>\n");
  fprintf(cont->vtufile, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\"");
  fprintf(cont->vtufile, " compressor=\"vtkZLibDataCompressor\"");
#ifdef SC_IS_BIGENDIAN
  fprintf(cont->vtufile, " byte_order=\"BigEndian\">\n");
#else
  fprintf(cont->vtufile, " byte_order=\"LittleEndian\">\n");
#endif
  fprintf(cont->vtufile, "  <UnstructuredGrid>\n");
  fprintf(cont->vtufile,
          "    <Piece NumberOfPoints=\"%lld\" NumberOfCells=\"%lld\">\n",
          (long long)Npoints, (long long)Ncells);
  fprintf(cont->vtufile, "      <Points>\n");

  float_data = P4EST_ALLOC(double, 3 * Npoints);

  /* write point position data */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"Position\""
                         " NumberOfComponents=\"3\" format=\"%s\">\n",
          cont->vtk_float_name, "binary");
  int base;
  int root = P8EST_ROOT_LEN;

  double xyz_quad[3], xyz_patch[3];
  double patch_offset;
  /* loop over the trees */
  for (jt = first_local_tree, quad_count = 0; jt <= last_local_tree; ++jt) {
    tree = p8est_tree_array_index(trees, jt);
    quadrants = &tree->quadrants;
    num_quads = quadrants->elem_count;

    /* loop over the elements in tree and calculate vertex coordinates */
    for (zz = 0; zz < num_quads; ++zz, ++quad_count) {
      quad = p8est_quadrant_array_index(quadrants, zz);
      base = P8EST_QUADRANT_LEN(quad->level);
      patch_offset = ((double)base / (LBADAPT_PATCHSIZE * (double)root));

      // lower left corner of quadrant
      lbadapt_get_front_lower_left(p8est, jt, quad, xyz_quad);

      patch_count = 0;
      for (int patch_z = 0; patch_z < LBADAPT_PATCHSIZE; ++patch_z) {
        for (int patch_y = 0; patch_y < LBADAPT_PATCHSIZE; ++patch_y) {
          for (int patch_x = 0; patch_x < LBADAPT_PATCHSIZE; ++patch_x) {
            // lower left corner of patch
            xyz_patch[0] = xyz_quad[0] + patch_x * patch_offset;
            xyz_patch[1] = xyz_quad[1] + patch_y * patch_offset;
            xyz_patch[2] = xyz_quad[2] + patch_z * patch_offset;
            k = 0;
            // calculate remaining coordinates
            for (zi = 0; zi < 2; ++zi) {
              for (yi = 0; yi < 2; ++yi) {
                for (xi = 0; xi < 2; ++xi) {
                  xyz[0] = xyz_patch[0] + xi * patch_offset;
                  xyz[1] = xyz_patch[1] + yi * patch_offset;
                  xyz[2] = xyz_patch[2] + zi * patch_offset;

                  for (j = 0; j < 3; ++j) {
                    float_data[j +
                               3 * (k +
                                    P8EST_CHILDREN *
                                        (patch_count +
                                         cells_per_patch * quad_count))] =
                        xyz[j];
                  }
                  ++k; // count coordinates written up to now
                }
              }
            }
            ++patch_count; // count patches written up to now
          }
        }
      }
    }
    assert(k == P8EST_CHILDREN);
    assert(patch_count == cells_per_patch);
  }
  assert(P8EST_CHILDREN * cells_per_patch * quad_count == Npoints);

  fprintf(cont->vtufile, "          ");
  /* TODO: Don't allocate the full size of the array, only allocate
   * the chunk that will be passed to zlib and do this a chunk
   * at a time.
   */
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)float_data,
                                   sizeof(*float_data) * 3 * Npoints);
  fprintf(cont->vtufile, "\n");
  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding points\n");
    lbadapt_vtk_context_destroy(cont);
    P4EST_FREE(float_data);
    return NULL;
  }
  P4EST_FREE(float_data);

  fprintf(cont->vtufile, "        </DataArray>\n");
  fprintf(cont->vtufile, "      </Points>\n");
  fprintf(cont->vtufile, "      <Cells>\n");

  /* write connectivity data */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"connectivity\""
                         " format=\"%s\">\n",
          P4EST_VTK_LOCIDX, "binary");

  fprintf(cont->vtufile, "          ");

  locidx_data = P4EST_ALLOC(p4est_locidx_t, Ncorners);
  for (il = 0; il < Ncorners; ++il) {
    locidx_data[il] = il;
  }
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                   sizeof(p4est_locidx_t) * Ncorners);
  P4EST_FREE(locidx_data);

  fprintf(cont->vtufile, "\n");
  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding connectivity\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");

  /* write offset data */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"offsets\""
                         " format=\"%s\">\n",
          P4EST_VTK_LOCIDX, "binary");
  locidx_data = P4EST_ALLOC(p4est_locidx_t, Ncells);
  for (il = 1; il <= Ncells; ++il) {
    locidx_data[il - 1] = P8EST_CHILDREN * il;
  }

  fprintf(cont->vtufile, "          ");
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                   sizeof(p4est_locidx_t) * Ncells);
  fprintf(cont->vtufile, "\n");

  P4EST_FREE(locidx_data);

  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding offsets\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");

  /* write type data */
  fprintf(cont->vtufile, "        <DataArray type=\"UInt8\" Name=\"types\""
                         " format=\"%s\">\n",
          "binary");
  uint8_data = P4EST_ALLOC(uint8_t, Ncells);
  for (il = 0; il < Ncells; ++il) {
    uint8_data[il] = P4EST_VTK_CELL_TYPE;
  }

  fprintf(cont->vtufile, "          ");
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)uint8_data,
                                   sizeof(*uint8_data) * Ncells);
  fprintf(cont->vtufile, "\n");

  P4EST_FREE(uint8_data);

  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");
  fprintf(cont->vtufile, "      </Cells>\n");

  if (ferror(cont->vtufile)) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error writing header\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }

  /* Only have the root write to the parallel vtk file */
  if (mpirank == 0) {
    snprintf(cont->pvtufilename, BUFSIZ, "%s.pvtu", filename);

    cont->pvtufile = fopen(cont->pvtufilename, "wb");
    if (!cont->pvtufile) {
      P4EST_LERRORF("Could not open %s for output\n", cont->pvtufilename);
      lbadapt_vtk_context_destroy(cont);
      return NULL;
    }

    fprintf(cont->pvtufile, "<?xml version=\"1.0\"?>\n");
    fprintf(cont->pvtufile,
            "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\"");
    fprintf(cont->pvtufile, " compressor=\"vtkZLibDataCompressor\"");
#ifdef SC_IS_BIGENDIAN
    fprintf(cont->pvtufile, " byte_order=\"BigEndian\">\n");
#else
    fprintf(cont->pvtufile, " byte_order=\"LittleEndian\">\n");
#endif

    fprintf(cont->pvtufile, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
    fprintf(cont->pvtufile, "    <PPoints>\n");
    fprintf(cont->pvtufile, "      <PDataArray type=\"%s\" Name=\"Position\""
                            " NumberOfComponents=\"3\" format=\"%s\"/>\n",
            cont->vtk_float_name, "binary");
    fprintf(cont->pvtufile, "    </PPoints>\n");

    if (ferror(cont->pvtufile)) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error writing parallel header\n");
      lbadapt_vtk_context_destroy(cont);
      return NULL;
    }

    /* Create a master file for visualization in Visit; this will be used
     * only in p4est_vtk_write_footer().
     */
    snprintf(cont->visitfilename, BUFSIZ, "%s.visit", filename);
    cont->visitfile = fopen(cont->visitfilename, "wb");
    if (!cont->visitfile) {
      P4EST_LERRORF("Could not open %s for output\n", cont->visitfilename);
      lbadapt_vtk_context_destroy(cont);
      return NULL;
    }
  }

  /* the nodes object is no longer needed */
  if (nodes != NULL) {
    p8est_nodes_destroy(cont->nodes);
    cont->nodes = NULL;
  }

  return cont;
}

lbadapt_vtk_context_t *
lbadapt_vtk_write_cell_scalar(lbadapt_vtk_context_t *cont,
                              const char *scalar_name, sc_array_t *values) {
  const p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  const p4est_locidx_t Ncells = cells_per_patch * p8est->local_num_quadrants;
  p4est_locidx_t il;
  int retval;
  double *float_data;

  P4EST_ASSERT(cont != NULL && cont->writing);

  /* Write cell data. */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"%s\""
                         " format=\"%s\">\n",
          cont->vtk_float_name, scalar_name, "binary");

  float_data = P4EST_ALLOC(double, Ncells);
  for (il = 0; il < Ncells; ++il) {
    float_data[il] = (double)*((double *)sc_array_index(values, il));
  }

  fprintf(cont->vtufile, "          ");
  /* TODO: Don't allocate the full size of the array, only allocate
   * the chunk that will be passed to zlib and do this a chunk
   * at a time.
   */
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)float_data,
                                   sizeof(*float_data) * Ncells);
  fprintf(cont->vtufile, "\n");

  P4EST_FREE(float_data);

  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding scalar cell data\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");

  if (ferror(cont->vtufile)) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error writing cell scalar file\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }

  return cont;
}

lbadapt_vtk_context_t *
lbadapt_vtk_write_cell_vector(lbadapt_vtk_context_t *cont,
                              const char *vector_name, sc_array_t *values) {
  const p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  const p4est_locidx_t Ncells = cells_per_patch * p8est->local_num_quadrants;
  p4est_locidx_t il;
  int retval;
  double *float_data;

  P4EST_ASSERT(cont != NULL && cont->writing);

  /* Write cell data. */
  fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"%s\""
                         " NumberOfComponents=\"3\" format=\"%s\">\n",
          cont->vtk_float_name, vector_name, "binary");

  float_data = P4EST_ALLOC(double, 3 * Ncells);
  for (il = 0; il < (3 * Ncells); ++il) {
    float_data[il] = (double)*((double *)sc_array_index(values, il));
  }

  fprintf(cont->vtufile, "          ");
  /* TODO: Don't allocate the full size of the array, only allocate
   * the chunk that will be passed to zlib and do this a chunk
   * at a time.
   */
  retval = sc_vtk_write_compressed(cont->vtufile, (char *)float_data,
                                   sizeof(*float_data) * 3 * Ncells);
  fprintf(cont->vtufile, "\n");

  P4EST_FREE(float_data);

  if (retval) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error encoding scalar cell data\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }
  fprintf(cont->vtufile, "        </DataArray>\n");

  if (ferror(cont->vtufile)) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error writing cell scalar file\n");
    lbadapt_vtk_context_destroy(cont);
    return NULL;
  }

  return cont;
}

/** Write VTK cell data.
 *
 * This function exports custom cell data to the vtk file; it is functionally
 * the same as \b p4est_vtk_write_cell_dataf with the only difference being
 * that instead of a variable argument list, an initialized \a va_list is
 * passed as the last argument. The \a va_list is initialized from the variable
 * argument list of the calling function.
 *
 * \note This function is actually called from \b p4est_vtk_write_cell_dataf
 * and does all of the work.
 *
 * \param [in,out] cont    A vtk context created by \ref p4est_vtk_context_new.
 * \param [in] num_cell_scalars Number of point scalar datasets to output.
 * \param [in] num_cell_vectors Number of point vector datasets to output.
 * \param [in,out] ap      An initialized va_list used to access the
 *                         scalar/vector data.
 *
 * \return          On success, the context that has been passed in.
 *                  On failure, returns NULL and deallocates the context.
 */
static lbadapt_vtk_context_t *
lbadapt_vtk_write_cell_datav(lbadapt_vtk_context_t *cont, int write_tree,
                             int write_level, int write_rank, int wrap_rank,
                             int write_qid, int num_cell_scalars,
                             int num_cell_vectors, va_list ap) {
  /* This function needs to do nothing if there is no data. */
  if (!(write_tree || write_level || write_rank || wrap_rank ||
        num_cell_vectors || num_cell_vectors))
    return cont;

  const int mpirank = p8est->mpirank;
  int retval;
  int i, all = 0;
  int scalar_strlen, vector_strlen;

  sc_array_t *trees = p8est->trees;
  p8est_tree_t *tree;
  const p4est_topidx_t first_local_tree = p8est->first_local_tree;
  const p4est_topidx_t last_local_tree = p8est->last_local_tree;

  const p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  const p4est_locidx_t Ncells = cells_per_patch * p8est->local_num_quadrants;

  char cell_scalars[BUFSIZ], cell_vectors[BUFSIZ];
  const char *name, **names;
  sc_array_t **values;
  size_t num_quads, zz;
  sc_array_t *quadrants;
  p8est_quadrant_t *quad;
  uint8_t *uint8_data;
  p4est_locidx_t *locidx_data;
  p4est_topidx_t jt;
  p4est_locidx_t il;

  P4EST_ASSERT(cont != NULL && cont->writing);
  P4EST_ASSERT(wrap_rank >= 0);

  values = P4EST_ALLOC(sc_array_t *, num_cell_scalars + num_cell_vectors);
  names = P4EST_ALLOC(const char *, num_cell_scalars + num_cell_vectors);

  /* Gather cell data. */
  scalar_strlen = 0;
  cell_scalars[0] = '\0';
  for (i = 0; i < num_cell_scalars; ++all, ++i) {
    name = names[all] = va_arg(ap, const char *);
    retval = snprintf(cell_scalars + scalar_strlen, BUFSIZ - scalar_strlen,
                      "%s%s", i == 0 ? "" : ",", name);
    SC_CHECK_ABORT(retval > 0,
                   P8EST_STRING "_vtk: Error collecting cell scalars");
    scalar_strlen += retval;
    values[all] = va_arg(ap, sc_array_t *);

    /* Validate input. */
    SC_CHECK_ABORT(values[all]->elem_size == sizeof(double),
                   P8EST_STRING "_vtk: Error: incorrect cell scalar data type; "
                                "scalar data must contain lb_floats.");
    SC_CHECK_ABORT(values[all]->elem_count == (size_t)Ncells,
                   P8EST_STRING "_vtk: Error: incorrect cell scalar data "
                                "count; scalar data must contain exactly "
                                "p4est->local_num_quadrants lb_floats.");
  }

  vector_strlen = 0;
  cell_vectors[0] = '\0';
  for (i = 0; i < num_cell_vectors; ++all, ++i) {
    name = names[all] = va_arg(ap, const char *);
    retval = snprintf(cell_vectors + vector_strlen, BUFSIZ - vector_strlen,
                      "%s%s", i == 0 ? "" : ",", name);
    SC_CHECK_ABORT(retval > 0,
                   P8EST_STRING "_vtk: Error collecting cell vectors");
    vector_strlen += retval;
    values[all] = va_arg(ap, sc_array_t *);

    /* Validate input. */
    SC_CHECK_ABORT(values[all]->elem_size == sizeof(double),
                   P8EST_STRING "_vtk: Error: incorrect cell vector data type; "
                                "vector data must contain lb_floats.");
    SC_CHECK_ABORT(values[all]->elem_count == (size_t)Ncells * 3,
                   P8EST_STRING "_vtk: Error: incorrect cell vector data "
                                "count; vector data must contain exactly "
                                "3*p4est->local_num_quadrants lb_floats.");
  }

  /* Check for pointer variable marking the end of variable data input. */
  lbadapt_vtk_context_t *end = va_arg(ap, lbadapt_vtk_context_t *);
  SC_CHECK_ABORT(
      end == cont, P8EST_STRING
      "_vtk Error: the end of variable "
      "data must be specified by passing, as the last argument, the current "
      "lbadapt_vtk_context_t struct.");

  char vtkCellDataString[BUFSIZ] = "";
  int printed = 0;

  if (write_tree)
    printed +=
        snprintf(vtkCellDataString + printed, BUFSIZ - printed, "treeid");

  if (write_level)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",level" : "level");

  if (write_rank)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",mpirank" : "mpirank");

  if (write_qid)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",qid" : "qid");

  if (num_cell_scalars)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",%s" : "%s", cell_scalars);

  if (num_cell_vectors)
    printed += snprintf(vtkCellDataString + printed, BUFSIZ - printed,
                        printed > 0 ? ",%s" : "%s", cell_vectors);

  fprintf(cont->vtufile, "      <CellData Scalars=\"%s\">\n",
          vtkCellDataString);

  locidx_data = P4EST_ALLOC(p4est_locidx_t, Ncells);
  uint8_data = P4EST_ALLOC(uint8_t, Ncells);

  if (write_tree) {
    fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"treeid\""
                           " format=\"%s\">\n",
            P4EST_VTK_LOCIDX, "binary");
    for (il = 0, jt = first_local_tree; jt <= last_local_tree; ++jt) {
      tree = p8est_tree_array_index(trees, jt);
      num_quads = tree->quadrants.elem_count;
      for (zz = 0; zz < num_quads; ++zz) {
        for (int p = 0; p < cells_per_patch; ++p, ++il) {
          locidx_data[il] = (p4est_locidx_t)jt;
        }
      }
    }
    fprintf(cont->vtufile, "          ");
    retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                     sizeof(*locidx_data) * Ncells);
    fprintf(cont->vtufile, "\n");
    if (retval) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(values);
      P4EST_FREE(names);
      P4EST_FREE(locidx_data);
      P4EST_FREE(uint8_data);

      return NULL;
    }
    fprintf(cont->vtufile, "        </DataArray>\n");
    P4EST_ASSERT(il == Ncells);
  }

  if (write_level) {
    fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"level\""
                           " format=\"%s\">\n",
            "UInt8", "binary");
    for (il = 0, jt = first_local_tree; jt <= last_local_tree; ++jt) {
      tree = p8est_tree_array_index(trees, jt);
      quadrants = &tree->quadrants;
      num_quads = quadrants->elem_count;
      for (zz = 0; zz < num_quads; ++zz) {
        quad = p8est_quadrant_array_index(quadrants, zz);
        for (int p = 0; p < cells_per_patch; ++p, ++il) {
          uint8_data[il] = (uint8_t)quad->level;
        }
      }
    }

    fprintf(cont->vtufile, "          ");
    retval = sc_vtk_write_compressed(cont->vtufile, (char *)uint8_data,
                                     sizeof(*uint8_data) * Ncells);
    fprintf(cont->vtufile, "\n");

    P4EST_FREE(uint8_data);

    if (retval) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(values);
      P4EST_FREE(names);
      P4EST_FREE(locidx_data);

      return NULL;
    }
    fprintf(cont->vtufile, "        </DataArray>\n");
  }

  if (write_rank) {
    const int wrapped_rank = wrap_rank > 0 ? mpirank % wrap_rank : mpirank;

    fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"mpirank\""
                           " format=\"%s\">\n",
            P4EST_VTK_LOCIDX, "binary");
    for (il = 0; il < Ncells; ++il)
      locidx_data[il] = (p4est_locidx_t)wrapped_rank;

    fprintf(cont->vtufile, "          ");
    retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                     sizeof(*locidx_data) * Ncells);
    fprintf(cont->vtufile, "\n");

    if (retval) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(values);
      P4EST_FREE(names);

      return NULL;
    }
    fprintf(cont->vtufile, "        </DataArray>\n");
  }

  if (write_qid) {
    fprintf(cont->vtufile, "        <DataArray type=\"%s\" Name=\"qid\""
                           " format=\"%s\">\n",
            P4EST_VTK_LOCIDX, "binary");
    int offset;
    for (il = 0, jt = first_local_tree; jt <= last_local_tree; ++jt) {
      if (0 == il) {
        offset = p8est->global_first_quadrant[mpirank];
      }
      else {
        offset = 0;
      }
      tree = p8est_tree_array_index(trees, jt);
      num_quads = tree->quadrants.elem_count;
      for (zz = 0; zz < num_quads; ++zz) {
        quad = p8est_quadrant_array_index(quadrants, zz);
        for (int p = 0; p < cells_per_patch; ++p, ++il) {
          locidx_data[il] = offset + (p4est_locidx_t)zz + tree->quadrants_offset;
        }
      }
    }
    fprintf(cont->vtufile, "          ");
    retval = sc_vtk_write_compressed(cont->vtufile, (char *)locidx_data,
                                     sizeof(*locidx_data) * Ncells);
    fprintf(cont->vtufile, "\n");

    P4EST_FREE(locidx_data);

    if (retval) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error encoding types\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(values);
      P4EST_FREE(names);
      P4EST_FREE(locidx_data);
      P4EST_FREE(uint8_data);

      return NULL;
    }
    fprintf(cont->vtufile, "        </DataArray>\n");
    P4EST_ASSERT(il == Ncells);
  }

  if (ferror(cont->vtufile)) {
    P4EST_LERRORF(P8EST_STRING "_vtk: Error writing %s\n", cont->vtufilename);
    lbadapt_vtk_context_destroy(cont);

    P4EST_FREE(values);
    P4EST_FREE(names);

    return NULL;
  }

  all = 0;
  for (i = 0; i < num_cell_scalars; ++all, ++i) {
    cont = lbadapt_vtk_write_cell_scalar(cont, names[all], values[all]);
    SC_CHECK_ABORT(cont != NULL,
                   P8EST_STRING "_vtk: Error writing cell scalars");
  }

  for (i = 0; i < num_cell_vectors; ++all, ++i) {
    cont = lbadapt_vtk_write_cell_vector(cont, names[all], values[all]);
    SC_CHECK_ABORT(cont != NULL,
                   P8EST_STRING "_vtk: Error writing cell vectors");
  }

  fprintf(cont->vtufile, "      </CellData>\n");

  P4EST_FREE(values);

  if (ferror(cont->vtufile)) {
    P4EST_LERRORF(P8EST_STRING "_vtk: Error writing %s\n", cont->vtufilename);
    lbadapt_vtk_context_destroy(cont);

    P4EST_FREE(names);

    return NULL;
  }

  /* Only have the root write to the parallel vtk file */
  if (mpirank == 0) {
    fprintf(cont->pvtufile, "    <PCellData Scalars=\"%s\">\n",
            vtkCellDataString);

    if (write_tree)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"treeid\" format=\"%s\"/>\n",
              P4EST_VTK_LOCIDX, "binary");

    if (write_level)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"level\" format=\"%s\"/>\n",
              "UInt8", "binary");

    if (write_rank)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"mpirank\" format=\"%s\"/>\n",
              P4EST_VTK_LOCIDX, "binary");

    if (write_qid)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"qid\" format=\"%s\"/>\n",
              P4EST_VTK_LOCIDX, "binary");

    all = 0;
    for (i = 0; i < num_cell_scalars; ++all, i++)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"%s\" format=\"%s\"/>\n",
              cont->vtk_float_name, names[all], "binary");

    for (i = 0; i < num_cell_vectors; ++all, i++)
      fprintf(cont->pvtufile,
              "      "
              "<PDataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"3\""
              " format=\"%s\"/>\n",
              cont->vtk_float_name, names[all], "binary");

    fprintf(cont->pvtufile, "    </PCellData>\n");

    if (ferror(cont->pvtufile)) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error writing parallel header\n");
      lbadapt_vtk_context_destroy(cont);

      P4EST_FREE(names);

      return NULL;
    }
  }

  P4EST_FREE(names);

  return cont;
}

lbadapt_vtk_context_t *
lbadapt_vtk_write_cell_dataf(lbadapt_vtk_context_t *cont, int write_tree,
                             int write_level, int write_rank, int wrap_rank,
                             int write_qid, int num_cell_scalars,
                             int num_cell_vectors, ...) {
  va_list ap;

  P4EST_ASSERT(cont != NULL && cont->writing);
  P4EST_ASSERT(num_cell_scalars >= 0 && num_cell_vectors >= 0);

  va_start(ap, num_cell_vectors);
  cont = lbadapt_vtk_write_cell_datav(cont, write_tree, write_level, write_rank,
                                      wrap_rank, write_qid, num_cell_scalars,
                                      num_cell_vectors, ap);
  va_end(ap);

  return cont;
}

int lbadapt_vtk_write_footer(lbadapt_vtk_context_t *cont) {
  int p;
  int procRank = p8est->mpirank;
  int numProcs = p8est->mpisize;

  P4EST_ASSERT(cont != NULL && cont->writing);

  fprintf(cont->vtufile, "    </Piece>\n");
  fprintf(cont->vtufile, "  </UnstructuredGrid>\n");
  fprintf(cont->vtufile, "</VTKFile>\n");

  if (ferror(cont->vtufile)) {
    P4EST_LERROR(P8EST_STRING "_vtk: Error writing footer\n");
    lbadapt_vtk_context_destroy(cont);
    return -1;
  }

  /* Only have the root write to the parallel vtk file */
  if (procRank == 0) {
    fprintf(cont->visitfile, "!NBLOCKS %d\n", numProcs);

    /* Do not write paths to parallel vtk file, because they will be in the same
     * directory as the vtk files written by each process */
    char *file_basename;
    file_basename = strrchr(cont->filename, '/');
    if (file_basename != NULL) {
      ++file_basename;
    } else {
      file_basename = cont->filename;
    }

    /* Write data about the parallel pieces into both files */
    for (p = 0; p < numProcs; ++p) {
      fprintf(cont->pvtufile, "    <Piece Source=\"%s_%04d.vtu\"/>\n",
              file_basename, p);
      fprintf(cont->visitfile, "%s_%04d.vtu\n", file_basename, p);
    }
    fprintf(cont->pvtufile, "  </PUnstructuredGrid>\n");
    fprintf(cont->pvtufile, "</VTKFile>\n");

    if (ferror(cont->pvtufile)) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error writing parallel footer\n");
      lbadapt_vtk_context_destroy(cont);
      return -1;
    }

    if (ferror(cont->visitfile)) {
      P4EST_LERROR(P8EST_STRING "_vtk: Error writing parallel footer\n");
      lbadapt_vtk_context_destroy(cont);
      return -1;
    }
  }

  /* Destroy context structure. */
  lbadapt_vtk_context_destroy(cont);

  return 0;
}

#endif // LB_ADAPTIVE_GPU

#endif // LB_ADAPTIVE
