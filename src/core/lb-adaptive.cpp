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
/** \file lb-boundaries.cpp
 *
 * Adaptive Lattice Boltzmann Scheme using CPU.
 * Implementation file for \ref lb-boundaries.hpp.
 *
 */

#include <stdlib.h>

#include "utils.hpp"
#include "constraint.hpp"
#include "communication.hpp"
#include "lb-adaptive.hpp"
#include "lb-boundaries.hpp"


#ifdef LB_ADAPTIVE

p8est_connectivity_t *conn;
p8est_t              *p8est;


/*** SETUP ***/
void lbadapt_init(p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant) {
  lbadapt_payload_t *data = (lbadapt_payload_t *) quadrant->p.user_data;

  data->boundary = 0;
  data->lbfields = malloc(sizeof(LB_FluidNode));
  data->lbfluid[0] = malloc(lbmodel.n_veloc * sizeof(double));
  data->lbfluid[1] = malloc(lbmodel.n_veloc * sizeof(double));
}


/*** REFINEMENT ***/
int refine_uniform (p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant) {
  return 1;
}


int refine_random (p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant) {
  return rand() % 2;
}


/*** HELPER FUNCTIONS ***/
void lbadapt_get_midpoint (p8est_t * p8est, p4est_topidx_t which_tree,
    p8est_quadrant_t * q, double xyz[3]) {
  p4est_qcoord_t half_length = P8EST_QUADRANT_LEN (q->level) * 0.5;

  p8est_qcoord_to_vertex (p8est->connectivity, which_tree,
                          q->x + half_length, q->y + half_length,
                          q->z + half_length,
                          xyz);
}


/*** ITERATOR CALLBACKS ***/
void lbadapt_get_boundary_status (p8est_iter_volume_info_t * info, void * user_data) {
  p8est_t * p8est = info->p4est;                                /* get p8est */
  p8est_quadrant_t * q = info->quad;                            /* get current global cell id */
  p4est_topidx_t which_tree = info->treeid;                     /* get current tree id */
  p4est_locidx_t local_id = info->quadid;                       /* get cell id w.r.t. tree-id */
  lbadapt_payload_t *data = (lbadapt_payload_t *) q->p.user_data; /* payload of cell */

  double midpoint[3];
  lbadapt_get_midpoint(p8est, which_tree, q, midpoint);

  data->boundary = lbadapt_is_boundary(midpoint);
}


void lbadapt_get_boundary_values (p8est_iter_volume_info_t * info, void * user_data) {
  double *bnd_vals = (double *) user_data;                      /* passed array to fill */
  p8est_t * p8est = info->p4est;                                /* get p8est */
  p8est_quadrant_t * q = info->quad;                            /* get current global cell id */
  p4est_topidx_t which_tree = info->treeid;                     /* get current tree id */
  p4est_locidx_t local_id = info->quadid;                       /* get cell id w.r.t. tree-id */
  p8est_tree_t * tree;
  lbadapt_payload_t *data = (lbadapt_payload_t *) q->p.user_data; /* payload of cell */

  double bnd;                                                     /* local meshwidth */
  p4est_locidx_t  arrayoffset;
  int i, j;

  tree = p8est_tree_array_index (p8est->trees, which_tree);
  local_id += tree->quadrants_offset;   /* now the id is relative to the MPI process */
  arrayoffset = local_id;      /* each local quadrant has 2^d (P4EST_CHILDREN) values in u_interp */

  /* just grab the u value of each cell and pass it into solution vector */
  bnd = data->boundary;
  bnd_vals[arrayoffset] = bnd;
}


void lbadapt_init_force_per_cell (p8est_iter_volume_info_t * info, void * user_data) {
  p8est_t           * p8est = info->p4est;                          /* get p8est */
  p8est_quadrant_t  * q     = info->quad;                           /* get current global cell id */
  lbadapt_payload_t * data  = (lbadapt_payload_t *) q->p.user_data; /* payload of cell */

  double h;                                                             /* local meshwidth */
  h = (double) P8EST_QUADRANT_LEN(q->level) / (double) P8EST_ROOT_LEN;

#ifdef EXTERNAL_FORCES
  // unit conversion: force density
  data->lbfields.force[0] = lbpar.ext_force[0] * SQR(h) * SQR(lbpar.tau);
  data->lbfields.force[1] = lbpar.ext_force[1] * SQR(h) * SQR(lbpar.tau);
  data->lbfields.force[2] = lbpar.ext_force[2] * SQR(h) * SQR(lbpar.tau);
#else // EXTERNAL_FORCES
  data->lbfields.force[0] = 0.0;
  data->lbfields.force[1] = 0.0;
  data->lbfields.force[2] = 0.0;
  data->lbfields.has_force = 0;
#endif // EXTERNAL_FORCES
}


void lbadapt_init_fluid_per_cell (p8est_iter_volume_info_t * info, void * user_data) {
  p8est_t * p8est = info->p4est;                                /* get p8est */
  p8est_quadrant_t * q = info->quad;                            /* get current global cell id */
  lbadapt_payload_t *data = (lbadapt_payload_t *) q->p.user_data; /* payload of cell */

  double h;                                                             /* local meshwidth */
  h = (double) P8EST_QUADRANT_LEN(q->level) / (double) P8EST_ROOT_LEN;

  // convert rho to lattice units
  double rho   = lbpar.rho[0] * h * h * h;
  // start with fluid at rest and no stress
  double j[3]  = {0., 0., 0.}
  double pi[6] = {0., 0., 0., 0., 0., 0.};
  lbadapt_calc_n_from_rho_j_pi (data->lbfields, rho, j, pi);
}
#endif // LB_ADAPTIVE
