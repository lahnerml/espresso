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


int lbadapt_calc_n_from_rho_j_pi (double * datafield,
                                  double rho,
                                  double * j,
                                  double * pi) {
  int i;
  double local_rho, local_j[3], local_pi[6], trace;
  const double avg_rho = lbpar.rho[0]*lbpar.agrid*lbpar.agrid*lbpar.agrid;

  local_rho  = rho;

  local_j[0] = j[0];
  local_j[1] = j[1];
  local_j[2] = j[2];

  for (i = 0; i < 6; i++) local_pi[i] = pi[i];

  trace = local_pi[0] + local_pi[2] + local_pi[5];

#ifdef D3Q19
  double rho_times_coeff;
  double tmp1,tmp2;

  /* update the q=0 sublattice */
  datafield[0][0] = 1./3. * (local_rho-avg_rho) - 1./2. * trace;

  /* update the q=1 sublattice */
  rho_times_coeff = 1./18. * (local_rho-avg_rho);

  datafield[0][1] = rho_times_coeff + 1./6.*local_j[0] + 1./4. * local_pi[0]
                    - 1./12.*trace;
  datafield[0][2] = rho_times_coeff - 1./6.*local_j[0] + 1./4. * local_pi[0]
                    - 1./12.*trace;
  datafield[0][3] = rho_times_coeff + 1./6.*local_j[1] + 1./4. * local_pi[2]
                    - 1./12.*trace;
  datafield[0][4] = rho_times_coeff - 1./6.*local_j[1] + 1./4. * local_pi[2]
                    - 1./12.*trace;
  datafield[0][5] = rho_times_coeff + 1./6.*local_j[2] + 1./4. * local_pi[5]
                    - 1./12.*trace;
  datafield[0][6] = rho_times_coeff - 1./6.*local_j[2] + 1./4. * local_pi[5]
                    - 1./12.*trace;

  /* update the q=2 sublattice */
  rho_times_coeff = 1./36. * (local_rho-avg_rho);

  tmp1 = local_pi[0] + local_pi[2];
  tmp2 = 2.0 * local_pi[1];

  datafield[0][7]  = rho_times_coeff + 1./12.*(local_j[0]+local_j[1])
                     + 1./8.*(tmp1+tmp2) - 1./24.*trace;
  datafield[0][8]  = rho_times_coeff - 1./12.*(local_j[0]+local_j[1])
                     + 1./8.*(tmp1+tmp2) - 1./24.*trace;
  datafield[0][9]  = rho_times_coeff + 1./12.*(local_j[0]-local_j[1])
                     + 1./8.*(tmp1-tmp2) - 1./24.*trace;
  datafield[0][10] = rho_times_coeff - 1./12.*(local_j[0]-local_j[1])
                     + 1./8.*(tmp1-tmp2) - 1./24.*trace;

  tmp1 = local_pi[0] + local_pi[5];
  tmp2 = 2.0 * local_pi[3];

  datafield[0][11] = rho_times_coeff + 1./12.*(local_j[0]+local_j[2])
                     + 1./8.*(tmp1+tmp2) - 1./24.*trace;
  datafield[0][12] = rho_times_coeff - 1./12.*(local_j[0]+local_j[2])
                     + 1./8.*(tmp1+tmp2) - 1./24.*trace;
  datafield[0][13] = rho_times_coeff + 1./12.*(local_j[0]-local_j[2])
                     + 1./8.*(tmp1-tmp2) - 1./24.*trace;
  datafield[0][14] = rho_times_coeff - 1./12.*(local_j[0]-local_j[2])
                     + 1./8.*(tmp1-tmp2) - 1./24.*trace;

  tmp1 = local_pi[2] + local_pi[5];
  tmp2 = 2.0 * local_pi[4];

  datafield[0][15] = rho_times_coeff + 1./12.*(local_j[1]+local_j[2])
                     + 1./8.*(tmp1+tmp2) - 1./24.*trace;
  datafield[0][16] = rho_times_coeff - 1./12.*(local_j[1]+local_j[2])
                     + 1./8.*(tmp1+tmp2) - 1./24.*trace;
  datafield[0][17] = rho_times_coeff + 1./12.*(local_j[1]-local_j[2])
                     + 1./8.*(tmp1-tmp2) - 1./24.*trace;
  datafield[0][18] = rho_times_coeff - 1./12.*(local_j[1]-local_j[2])
                     + 1./8.*(tmp1-tmp2) - 1./24.*trace;
#else // D3Q19
  int i;
  double tmp=0.0;
  double (*c)[3] = lbmodel.c;
  double (*coeff)[4] = lbmodel.coeff;

  for (i = 0; i < lbmodel.n_veloc; i++) {
    tmp = local_pi[0] * SQR(c[i][0])
      + (2.0 * local_pi[1] * c[i][0] + local_pi[2] * c[i][1])*c[i][1]
      + (2.0 * (local_pi[3]*c[i][0] + local_pi[4] * c[i][1])
         + local_pi[5] * c[i][2]) * c[i][2];

    datafield[0][i] =  coeff[i][0] * (local_rho-avg_rho);
    datafield[0][i] += coeff[i][1] * scalar(local_j,c[i]);
    datafield[0][i] += coeff[i][2] * tmp;
    datafield[0][i] += coeff[i][3] * trace;
  }
#endif // D3Q19

  return 0;
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

  double bnd;
  p4est_locidx_t  arrayoffset;

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

  double h;                                                         /* local meshwidth */
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


void lbadapt_calc_local_rho (p8est_iter_volume_info_t * info, void * user_data) {
  double *rho = (double *) user_data;                           /* passed double to fill */
  p8est_t * p8est = info->p4est;                                /* get p8est */
  p8est_quadrant_t * q = info->quad;                            /* get current global cell id */
  p4est_topidx_t which_tree = info->treeid;                     /* get current tree id */
  p4est_locidx_t local_id = info->quadid;                       /* get cell id w.r.t. tree-id */
  lbadapt_payload_t *data = (lbadapt_payload_t *) q->p.user_data; /* payload of cell */

  double h;                                                     /* local meshwidth */
  h = (double) P8EST_QUADRANT_LEN(q->level) / (double) P8EST_ROOT_LEN;

#ifndef D3Q19
#error Only D3Q19 is implemened!
#endif // D3Q19

  // unit conversion: mass density
  if (!(lattice_switch & LATTICE_LB)) {
      ostringstream msg;
      msg <<"Error in lb_calc_local_rho in " << __FILE__ << __LINE__ << ": CPU LB not switched on.";
      runtimeError(msg);
    *rho =0;
    return;
  }

  double avg_rho = lbpar.rho[0] * h * h * h;

  *rho +=   avg_rho
          + data->lbfluid[0][0]
          + data->lbfluid[0][1]  + data->lbfluid[0][2]
          + data->lbfluid[0][3]  + data->lbfluid[0][4]
          + data->lbfluid[0][5]  + data->lbfluid[0][6]
          + data->lbfluid[0][7]  + data->lbfluid[0][8]
          + data->lbfluid[0][9]  + data->lbfluid[0][10]
          + data->lbfluid[0][11] + data->lbfluid[0][12]
          + data->lbfluid[0][13] + data->lbfluid[0][14]
          + data->lbfluid[0][15] + data->lbfluid[0][16]
          + data->lbfluid[0][17] + data->lbfluid[0][18];
}


void lbadapt_calc_local_j (p8est_iter_volume_info_t * info, void *user_data) {
  double *momentum = (double *) user_data;                      /* passed array to fill */
  p8est_t * p8est = info->p4est;                                /* get p8est */
  p8est_quadrant_t * q = info->quad;                            /* get current global cell id */
  lbadapt_payload_t *data = (lbadapt_payload_t *) q->p.user_data; /* payload of cell */
  double h;                                                     /* local meshwidth */
  h = (double) P8EST_QUADRANT_LEN(q->level) / (double) P8EST_ROOT_LEN;

  double j[3];

#ifndef D3Q19
#error Only D3Q19 is implemened!
#endif // D3Q19
  if (!(lattice_switch & LATTICE_LB)) {
    ostringstream msg;
    msg <<"Error in lb_calc_local_j in " << __FILE__ << __LINE__ << ": CPU LB not switched on.";
    runtimeError(msg);
    j[0]=j[1]=j[2]=0;
    return;
  }

  j[0] =   data->lbfluid[0][1]  - data->lbfluid[0][2]
         + data->lbfluid[0][7]  - data->lbfluid[0][8]
         + data->lbfluid[0][9]  - data->lbfluid[0][10]
         + data->lbfluid[0][11] - data->lbfluid[0][12]
         + data->lbfluid[0][13] - data->lbfluid[0][14];
  j[1] =   data->lbfluid[0][3]  - data->lbfluid[0][4]
         + data->lbfluid[0][7]  - data->lbfluid[0][8]
         - data->lbfluid[0][9]  + data->lbfluid[0][10]
         + data->lbfluid[0][15] - data->lbfluid[0][16]
         + data->lbfluid[0][17] - data->lbfluid[0][18];
  j[2] =   data->lbfluid[0][5]  - data->lbfluid[0][6]
         + data->lbfluid[0][11] - data->lbfluid[0][12]
         - data->lbfluid[0][13] + data->lbfluid[0][14]
         + data->lbfluid[0][15] - data->lbfluid[0][16]
         - data->lbfluid[0][17] + data->lbfluid[0][18];
  momentum[0] += j[0] + lbfields.force[0];
  momentum[1] += j[1] + lbfields.force[1];
  momentum[2] += j[2] + lbfields.force[2];

  momentum[0] *= h/lbpar.tau;
  momentum[1] *= h/lbpar.tau;
  momentum[2] *= h/lbpar.tau;
}


void lbadapt_calc_local_pi (p8est_iter_volume_info_t * info, void *user_data) {
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
#endif // LB_ADAPTIVE
