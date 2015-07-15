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
#include <iostream>

#include "utils.hpp"
#include "constraint.hpp"
#include "communication.hpp"
#include "lb.hpp"
#include "lb-adaptive.hpp"
#include "lb-boundaries.hpp"
#include "lb-d3q19.hpp"

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
#endif // (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) && !defined(GAUSSRANDOM))


/* "external variables" */
p8est_connectivity_t *conn;
p8est_t              *p8est;
p8est_ghost_t        *lbadapt_ghost;
p8est_mesh_t         *lbadapt_mesh;
lbadapt_payload_t    *lbadapt_ghost_data;

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
void lbadapt_init(p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant) {
  lbadapt_payload_t *data = (lbadapt_payload_t *) quadrant->p.user_data;

  data->boundary = 0;
  // data->lbfields = (LB_FluidNode) malloc(sizeof(LB_FluidNode));
  data->lbfluid[0] = (double*) malloc(lbmodel.n_veloc * sizeof(double));
  data->lbfluid[1] = (double*) malloc(lbmodel.n_veloc * sizeof(double));
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


int lbadapt_calc_n_from_rho_j_pi (double ** datafield,
                                  double rho,
                                  double * j,
                                  double * pi,
                                  double h) {
  int i;
  double local_rho, local_j[3], local_pi[6], trace;
  const double avg_rho = lbpar.rho[0] * h * h * h;

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
  datafield[0][0] = 1./3. * (local_rho-avg_rho) - 0.5 * trace;

  /* update the q=1 sublattice */
  rho_times_coeff = 1./18. * (local_rho-avg_rho);

  datafield[0][1] = rho_times_coeff + 1./6.*local_j[0] + 0.25 * local_pi[0]
                    - 1./12.*trace;
  datafield[0][2] = rho_times_coeff - 1./6.*local_j[0] + 0.25 * local_pi[0]
                    - 1./12.*trace;
  datafield[0][3] = rho_times_coeff + 1./6.*local_j[1] + 0.25 * local_pi[2]
                    - 1./12.*trace;
  datafield[0][4] = rho_times_coeff - 1./6.*local_j[1] + 0.25 * local_pi[2]
                    - 1./12.*trace;
  datafield[0][5] = rho_times_coeff + 1./6.*local_j[2] + 0.25 * local_pi[5]
                    - 1./12.*trace;
  datafield[0][6] = rho_times_coeff - 1./6.*local_j[2] + 0.25 * local_pi[5]
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


int lbadapt_calc_modes(double ** population, double * mode) {
#ifdef D3Q19
  double n0, n1p, n1m, n2p, n2m, n3p, n3m, n4p, n4m, n5p, n5m, n6p, n6m, \
    n7p, n7m, n8p, n8m, n9p, n9m;

  n0  = population[0][0];
  n1p = population[0][1] + population[0][2];
  n1m = population[0][1] - population[0][2];
  n2p = population[0][3] + population[0][4];
  n2m = population[0][3] - population[0][4];
  n3p = population[0][5] + population[0][6];
  n3m = population[0][5] - population[0][6];
  n4p = population[0][7] + population[0][8];
  n4m = population[0][7] - population[0][8];
  n5p = population[0][9] + population[0][10];
  n5m = population[0][9] - population[0][10];
  n6p = population[0][11] + population[0][12];
  n6m = population[0][11] - population[0][12];
  n7p = population[0][13] + population[0][14];
  n7m = population[0][13] - population[0][14];
  n8p = population[0][15] + population[0][16];
  n8m = population[0][15] - population[0][16];
  n9p = population[0][17] + population[0][18];
  n9m = population[0][17] - population[0][18];

  /* mass mode */
  mode[0] = n0 + n1p + n2p + n3p + n4p + n5p + n6p + n7p + n8p + n9p;

  /* momentum modes */
  mode[1] = n1m + n4m + n5m + n6m + n7m;
  mode[2] = n2m + n4m - n5m + n8m + n9m;
  mode[3] = n3m + n6m - n7m + n8m - n9m;

  /* stress modes */
  mode[4] = -n0 + n4p + n5p + n6p + n7p + n8p + n9p;
  mode[5] = n1p - n2p + n6p + n7p - n8p - n9p;
  mode[6] = n1p + n2p - n6p - n7p - n8p - n9p - 2.*(n3p - n4p - n5p);
  mode[7] = n4p - n5p;
  mode[8] = n6p - n7p;
  mode[9] = n8p - n9p;

#ifndef OLD_FLUCT
  /* kinetic modes */
  mode[10] = -2.*n1m + n4m + n5m + n6m + n7m;
  mode[11] = -2.*n2m + n4m - n5m + n8m + n9m;
  mode[12] = -2.*n3m + n6m - n7m + n8m - n9m;
  mode[13] = n4m + n5m - n6m - n7m;
  mode[14] = n4m - n5m - n8m - n9m;
  mode[15] = n6m - n7m - n8m + n9m;
  mode[16] = n0 + n4p + n5p + n6p + n7p + n8p + n9p
    - 2.*(n1p + n2p + n3p);
  mode[17] = - n1p + n2p + n6p + n7p - n8p - n9p;
  mode[18] = - n1p - n2p -n6p - n7p - n8p - n9p
    + 2.*(n3p + n4p + n5p);
#endif // !OLD_FLUCT

#else // D3Q19
  int i, j;
  for (i = 0; i < lbmodel.n_veloc; i++) {
    mode[i] = 0.0;
    for (j = 0; j < lbmodel.n_veloc; j++) {
      mode[i] += lbmodel.e[i][j] * lbfluid[0][i][index];
    }
  }
#endif // D3Q19

  return 0;
}


int lbadapt_relax_modes (double * mode, double * force, double h) {
  double rho, j[3], pi_eq[6];

  /* re-construct the real density
   * remember that the populations are stored as differences to their
   * equilibrium value */
  rho = mode[0] + lbpar.rho[0] * h * h * h;

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
  pi_eq[0] = scalar(j,j) / rho;
  pi_eq[1] = (SQR(j[0])-SQR(j[1])) / rho;
  pi_eq[2] = (scalar(j,j) - 3.0 * SQR(j[2])) / rho;
  pi_eq[3] = j[0] * j[1] / rho;
  pi_eq[4] = j[0] * j[2] / rho;
  pi_eq[5] = j[1] * j[2] / rho;

  /* relax the stress modes */
  mode[4] = pi_eq[0] + gamma_bulk * (mode[4] - pi_eq[0]);
  mode[5] = pi_eq[1] + gamma_shear * (mode[5] - pi_eq[1]);
  mode[6] = pi_eq[2] + gamma_shear * (mode[6] - pi_eq[2]);
  mode[7] = pi_eq[3] + gamma_shear * (mode[7] - pi_eq[3]);
  mode[8] = pi_eq[4] + gamma_shear * (mode[8] - pi_eq[4]);
  mode[9] = pi_eq[5] + gamma_shear * (mode[9] - pi_eq[5]);

#ifndef OLD_FLUCT
  /* relax the ghost modes (project them out) */
  /* ghost modes have no equilibrium part due to orthogonality */
  mode[10] = gamma_odd*mode[10];
  mode[11] = gamma_odd*mode[11];
  mode[12] = gamma_odd*mode[12];
  mode[13] = gamma_odd*mode[13];
  mode[14] = gamma_odd*mode[14];
  mode[15] = gamma_odd*mode[15];
  mode[16] = gamma_even*mode[16];
  mode[17] = gamma_even*mode[17];
  mode[18] = gamma_even*mode[18];
#endif // !OLD_FLUCT

  return 0;
}


int lbadapt_thermalize_modes(double * mode, double h) {
  double fluct[6];
#ifdef GAUSSRANDOM
  double rootrho_gauss = sqrt(fabs(mode[0]+lbpar.rho[0] * h * h * h));

  /* stress modes */
  mode[4] += (fluct[0] = rootrho_gauss*lb_phi[4]*gaussian_random());
  mode[5] += (fluct[1] = rootrho_gauss*lb_phi[5]*gaussian_random());
  mode[6] += (fluct[2] = rootrho_gauss*lb_phi[6]*gaussian_random());
  mode[7] += (fluct[3] = rootrho_gauss*lb_phi[7]*gaussian_random());
  mode[8] += (fluct[4] = rootrho_gauss*lb_phi[8]*gaussian_random());
  mode[9] += (fluct[5] = rootrho_gauss*lb_phi[9]*gaussian_random());

#ifndef OLD_FLUCT
  /* ghost modes */
  mode[10] += rootrho_gauss*lb_phi[10]*gaussian_random();
  mode[11] += rootrho_gauss*lb_phi[11]*gaussian_random();
  mode[12] += rootrho_gauss*lb_phi[12]*gaussian_random();
  mode[13] += rootrho_gauss*lb_phi[13]*gaussian_random();
  mode[14] += rootrho_gauss*lb_phi[14]*gaussian_random();
  mode[15] += rootrho_gauss*lb_phi[15]*gaussian_random();
  mode[16] += rootrho_gauss*lb_phi[16]*gaussian_random();
  mode[17] += rootrho_gauss*lb_phi[17]*gaussian_random();
  mode[18] += rootrho_gauss*lb_phi[18]*gaussian_random();
#endif // !OLD_FLUCT

#elif defined (GAUSSRANDOMCUT)
  double rootrho_gauss = sqrt(fabs(mode[0]+lbpar.rho[0] * h * h * h));

  /* stress modes */
  mode[4] += (fluct[0] = rootrho_gauss*lb_phi[4]*gaussian_random_cut());
  mode[5] += (fluct[1] = rootrho_gauss*lb_phi[5]*gaussian_random_cut());
  mode[6] += (fluct[2] = rootrho_gauss*lb_phi[6]*gaussian_random_cut());
  mode[7] += (fluct[3] = rootrho_gauss*lb_phi[7]*gaussian_random_cut());
  mode[8] += (fluct[4] = rootrho_gauss*lb_phi[8]*gaussian_random_cut());
  mode[9] += (fluct[5] = rootrho_gauss*lb_phi[9]*gaussian_random_cut());

#ifndef OLD_FLUCT
  /* ghost modes */
  mode[10] += rootrho_gauss*lb_phi[10]*gaussian_random_cut();
  mode[11] += rootrho_gauss*lb_phi[11]*gaussian_random_cut();
  mode[12] += rootrho_gauss*lb_phi[12]*gaussian_random_cut();
  mode[13] += rootrho_gauss*lb_phi[13]*gaussian_random_cut();
  mode[14] += rootrho_gauss*lb_phi[14]*gaussian_random_cut();
  mode[15] += rootrho_gauss*lb_phi[15]*gaussian_random_cut();
  mode[16] += rootrho_gauss*lb_phi[16]*gaussian_random_cut();
  mode[17] += rootrho_gauss*lb_phi[17]*gaussian_random_cut();
  mode[18] += rootrho_gauss*lb_phi[18]*gaussian_random_cut();
#endif // OLD_FLUCT

#elif defined (FLATNOISE)
  double rootrho = sqrt(fabs(12.0*(mode[0]+lbpar.rho[0] * h * h * h)));

  /* stress modes */
  mode[4] += (fluct[0] = rootrho*lb_phi[4]*(d_random()-0.5));
  mode[5] += (fluct[1] = rootrho*lb_phi[5]*(d_random()-0.5));
  mode[6] += (fluct[2] = rootrho*lb_phi[6]*(d_random()-0.5));
  mode[7] += (fluct[3] = rootrho*lb_phi[7]*(d_random()-0.5));
  mode[8] += (fluct[4] = rootrho*lb_phi[8]*(d_random()-0.5));
  mode[9] += (fluct[5] = rootrho*lb_phi[9]*(d_random()-0.5));

#ifndef OLD_FLUCT
  /* ghost modes */
  mode[10] += rootrho*lb_phi[10]*(d_random()-0.5);
  mode[11] += rootrho*lb_phi[11]*(d_random()-0.5);
  mode[12] += rootrho*lb_phi[12]*(d_random()-0.5);
  mode[13] += rootrho*lb_phi[13]*(d_random()-0.5);
  mode[14] += rootrho*lb_phi[14]*(d_random()-0.5);
  mode[15] += rootrho*lb_phi[15]*(d_random()-0.5);
  mode[16] += rootrho*lb_phi[16]*(d_random()-0.5);
  mode[17] += rootrho*lb_phi[17]*(d_random()-0.5);
  mode[18] += rootrho*lb_phi[18]*(d_random()-0.5);
#endif // !OLD_FLUCT
#else // GAUSSRANDOM
#error No noise type defined for the CPU LB
#endif //GAUSSRANDOM

#ifdef ADDITIONAL_CHECKS
  rancounter += 15;
#endif // ADDITIONAL_CHECKS

  return 0;
}


int lbadapt_apply_forces (double * mode, LB_FluidNode * lbfields, double h) {
  double rho, u[3], C[6], *f;

  f = lbfields->force;

  rho = mode[0] + lbpar.rho[0] * h * h * h;

  /* hydrodynamic momentum density is redefined when external forces present */
  u[0] = (mode[1] + 0.5 * f[0])/rho;
  u[1] = (mode[2] + 0.5 * f[1])/rho;
  u[2] = (mode[3] + 0.5 * f[2])/rho;

  C[0] = (1.+gamma_bulk)*u[0]*f[0] + 1./3.*(gamma_bulk-gamma_shear)*scalar(u,f);
  C[2] = (1.+gamma_bulk)*u[1]*f[1] + 1./3.*(gamma_bulk-gamma_shear)*scalar(u,f);
  C[5] = (1.+gamma_bulk)*u[2]*f[2] + 1./3.*(gamma_bulk-gamma_shear)*scalar(u,f);
  C[1] = 1./2. * (1.+gamma_shear)*(u[0]*f[1]+u[1]*f[0]);
  C[3] = 1./2. * (1.+gamma_shear)*(u[0]*f[2]+u[2]*f[0]);
  C[4] = 1./2. * (1.+gamma_shear)*(u[1]*f[2]+u[2]*f[1]);

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

  /* reset force */
#ifdef EXTERNAL_FORCES
  // unit conversion: force density
  lbfields->force[0] = lbpar.ext_force[0] * SQR(h) * SQR(lbpar.tau);
  lbfields->force[1] = lbpar.ext_force[1] * SQR(h) * SQR(lbpar.tau);
  lbfields->force[2] = lbpar.ext_force[2] * SQR(h) * SQR(lbpar.tau);
#else // EXTERNAL_FORCES
  lbfields->force[0] = 0.0;
  lbfields->force[1] = 0.0;
  lbfields->force[2] = 0.0;
  lbfields->has_force = 0;
#endif // EXTERNAL_FORCES

  return 0;
}


int lbadapt_calc_pop_from_modes (double ** populations, double * mode) {
  double *w = lbmodel.w;

#ifdef D3Q19
  double (*e)[19] = d3q19_modebase;
  double m[19];

  /* normalization factors enter in the back transformation */
  for (int i = 0; i < lbmodel.n_veloc; i++) {
    m[i] = (1./e[19][i])*mode[i];
  }

  populations[0][ 0] = m[0] - m[4] + m[16];
  populations[0][ 1] = m[0] + m[1] + m[5] + m[6] - m[17] - m[18]
                       - 2.*(m[10] + m[16]);
  populations[0][ 2] = m[0] - m[1] + m[5] + m[6] - m[17] - m[18]
                       + 2.*(m[10] - m[16]);
  populations[0][ 3] = m[0] + m[2] - m[5] + m[6] + m[17] - m[18]
                       - 2.*(m[11] + m[16]);
  populations[0][ 4] = m[0] - m[2] - m[5] + m[6] + m[17] - m[18]
                       + 2.*(m[11] - m[16]);
  populations[0][ 5] = m[0] + m[3] - 2.*(m[6] + m[12] + m[16] - m[18]);
  populations[0][ 6] = m[0] - m[3] - 2.*(m[6] - m[12] + m[16] - m[18]);
  populations[0][ 7] = m[0] + m[ 1] + m[ 2] + m[ 4] + 2.*m[6] + m[7]
                       + m[10] + m[11] + m[13] + m[14] + m[16] + 2.*m[18];
  populations[0][ 8] = m[0] - m[ 1] - m[ 2] + m[ 4] + 2.*m[6] + m[7]
                       - m[10] - m[11] - m[13] - m[14] + m[16] + 2.*m[18];
  populations[0][ 9] = m[0] + m[ 1] - m[ 2] + m[ 4] + 2.*m[6] - m[7]
                       + m[10] - m[11] + m[13] - m[14] + m[16] + 2.*m[18];
  populations[0][10] = m[0] - m[ 1] + m[ 2] + m[ 4] + 2.*m[6] - m[7]
                       - m[10] + m[11] - m[13] + m[14] + m[16] + 2.*m[18];
  populations[0][11] = m[0] + m[ 1] + m[ 3] + m[ 4] + m[ 5] - m[ 6]
                       + m[8] + m[10] + m[12] - m[13] + m[15] + m[16]
                       + m[17] - m[18];
  populations[0][12] = m[0] - m[ 1] - m[ 3] + m[ 4] + m[ 5] - m[ 6]
                       + m[8] - m[10] - m[12] + m[13] - m[15] + m[16]
                       + m[17] - m[18];
  populations[0][13] = m[0] + m[ 1] - m[ 3] + m[ 4] + m[ 5] - m[ 6]
                       - m[8] + m[10] - m[12] - m[13] - m[15] + m[16]
                       + m[17] - m[18];
  populations[0][14] = m[0] - m[ 1] + m[ 3] + m[ 4] + m[ 5] - m[ 6]
                       - m[8] - m[10] + m[12] + m[13] + m[15] + m[16]
                       + m[17] - m[18];
  populations[0][15] = m[0] + m[ 2] + m[ 3] + m[ 4] - m[ 5] - m[ 6]
                       + m[9] + m[11] + m[12] - m[14] - m[15] + m[16]
                       - m[17] - m[18];
  populations[0][16] = m[0] - m[ 2] - m[ 3] + m[ 4] - m[ 5] - m[ 6]
                       + m[9] - m[11] - m[12] + m[14] + m[15] + m[16]
                       - m[17] - m[18];
  populations[0][17] = m[0] + m[ 2] - m[ 3] + m[ 4] - m[ 5] - m[ 6]
                       - m[9] + m[11] - m[12] - m[14] + m[15] + m[16]
                       - m[17] - m[18];
  populations[0][18] = m[0] - m[ 2] + m[ 3] + m[ 4] - m[ 5] - m[ 6]
                       - m[9] - m[11] + m[12] + m[14] - m[15] + m[16]
                       - m[17] - m[18];

  /* weights enter in the back transformation */
  for (int i = 0; i < lbmodel.n_veloc; i++) {
    populations[0][i] *= w[i];
  }

#else // D3Q19
  double **e = lbmodel.e;
  for (int i = 0; i < lbmodel.n_veloc; i++) {
    populations[0][i] = 0.0;

    for (int j = 0; j < lbmodel.n_veloc; j++) {
      populations[0][i] += mode[j] * e[j][i] / e[19][j];
    }

    populations[0][i] *= w[i];
  }
#endif // D3Q19
  return 0;
}


int lbadapt_calc_n_from_modes_push (int qid, double * m) {
#ifdef D3Q19
  /* indexes of streaming targets */
  /* currently:
   * no tree boundaries, i.e. no need to consider orientation.
   * direct transfer from mapping given at beginning of file */
  index_t next[19];
  /* 0 vel */
  next[0]  = qid;
  /* face-velocities */
  next[1]  = lbadapt_mesh->quad_to_quad[P8EST_FACES * qid + 1];
  next[2]  = lbadapt_mesh->quad_to_quad[P8EST_FACES * qid + 0];
  next[3]  = lbadapt_mesh->quad_to_quad[P8EST_FACES * qid + 3];
  next[4]  = lbadapt_mesh->quad_to_quad[P8EST_FACES * qid + 2];
  next[5]  = lbadapt_mesh->quad_to_quad[P8EST_FACES * qid + 5];
  next[6]  = lbadapt_mesh->quad_to_quad[P8EST_FACES * qid + 4];
  /* edge velocities */
  next[7]  = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid + 11];
  next[8]  = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  8];
  next[9]  = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  9];
  next[10] = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid + 10];
  next[11] = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  7];
  next[12] = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  4];
  next[13] = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  5];
  next[14] = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  6];
  next[15] = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  3];
  next[16] = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  0];
  next[17] = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  1];
  next[18] = lbadapt_mesh->quad_to_quad_edge[P8EST_EDGES * qid +  2];

  /* normalization factors enter in the back transformation */
  for (int i = 0; i < lbmodel.n_veloc; i++) {
    m[i] = (1. / d3q19_modebase[19][i]) * m[i];
  }

  p8est_tree_t      * tree;
  p8est_quadrant_t  * cell;
  lbadapt_payload_t * data;
#ifndef OLD_FLUCT
  tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
  cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[0]);
  data = (lbadapt_payload_t *) cell->p.user_data;
  data->lbfluid[1][0] = (m[0] - m[4] + m[16]) * lbmodel.w[0];

  if (next[1] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 1] = (m[0] + m[1] + m[5] + m[6] - m[17] - m[18]
        - 2.*(m[10] + m[16])) * lbmodel.w[1];
  }

  if (next[2] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[2]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 2] = (m[0] - m[1] + m[5] + m[6] - m[17] - m[18]
        + 2.*(m[10] - m[16])) * lbmodel.w[2];
  }

  if (next[3] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[3]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 3] = (m[0] + m[2] - m[5] + m[6] + m[17] - m[18]
        - 2.*(m[11] + m[16])) * lbmodel.w[3];
  }

  if (next[4] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[4]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 4] = (m[0] - m[2] - m[5] + m[6] + m[17] - m[18]
        + 2.*(m[11] - m[16])) * lbmodel.w[4];
  }

  if (next[5] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[5]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 5] = (m[0] + m[3] - 2.*(m[6] + m[12] + m[16] - m[18])) * lbmodel.w[5];
  }

  if (next[6] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[6]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 6] = (m[0] - m[3] - 2.*(m[6] - m[12] + m[16] - m[18])) * lbmodel.w[6];
  }

  if (next[7] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[7]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 7] = (m[0] + m[1] + m[2] + m[4] + 2.*m[6] + m[7]
        + m[10] + m[11] + m[13] + m[14] + m[16] + 2.*m[18]) * lbmodel.w[7];
  }

  if (next[8] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[8]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 8] = (m[0] - m[1] - m[2] + m[4] + 2.*m[6] + m[7]
        - m[10] - m[11] - m[13] - m[14] + m[16] + 2.*m[18]) * lbmodel.w[8];
  }

  if (next[9] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[9]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 9] = (m[0] + m[1] - m[2] + m[4] + 2.*m[6] - m[7]
        + m[10] - m[11] + m[13] - m[14] + m[16] + 2.*m[18]) * lbmodel.w[9];
  }

  if (next[10] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[10]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][10] = (m[0] - m[1] + m[2] + m[4] + 2.*m[6] - m[7]
        - m[10] + m[11] - m[13] + m[14] + m[16] + 2.*m[18]) * lbmodel.w[10];
  }

  if (next[11] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[11]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][11] = (m[0] + m[1] + m[3] + m[4] + m[5] - m[6] + m[8]
        + m[10] + m[12] - m[13] + m[15] + m[16] + m[17] - m[18]) * lbmodel.w[11];
  }

  if (next[12] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[12]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][12] = (m[0] - m[1] - m[3] + m[4] + m[5] - m[6] + m[8]
        - m[10] - m[12] + m[13] - m[15] + m[16] + m[17] - m[18]) * lbmodel.w[12];
  }

  if (next[13] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[13]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][13] = (m[0] + m[1] - m[3] + m[4] + m[5] - m[6] - m[8]
        + m[10] - m[12] - m[13] - m[15] + m[16] + m[17] - m[18]) * lbmodel.w[13];
  }

  if (next[14] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[14]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][14] = (m[0] - m[1] + m[3] + m[4] + m[5] - m[6] - m[8]
        - m[10] + m[12] + m[13] + m[15] + m[16] + m[17] - m[18]) * lbmodel.w[14];
  }

  if (next[15] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[15]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][15] = (m[0] + m[2] + m[3] + m[4] - m[5] - m[6] + m[9]
        + m[11] + m[12] - m[14] - m[15] + m[16] - m[17] - m[18]) * lbmodel.w[15];
  }

  if (next[16] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[16]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][16] = (m[0] - m[2] - m[3] + m[4] - m[5] - m[6] + m[9]
        - m[11] - m[12] + m[14] + m[15] + m[16] - m[17] - m[18]) * lbmodel.w[16];
  }

  if (next[17] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[17]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][17] = (m[0] + m[2] - m[3] + m[4] - m[5] - m[6] - m[9]
        + m[11] - m[12] - m[14] + m[15] + m[16] - m[17] - m[18]) * lbmodel.w[17];
  }

  if (next[18] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[18]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][18] = (m[0] - m[2] + m[3] + m[4] - m[5] - m[6] - m[9]
        - m[11] + m[12] + m[14] - m[15] + m[16] - m[17] - m[18]) * lbmodel.w[18];
  }
#else // !OLD_FLUCT
  tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
  cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
  data = (lbadapt_payload_t *) cell->p.user_data;
  data->lbfluid[1][ 0] = (m[0] - m[4]) * lbmodel.w[0];

  if (next[1] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 1] = (m[0] + m[1] + m[5] + m[6]) * lbmodel.w[1];
  }

  if (next[2] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 2] = (m[0] - m[1] + m[5] + m[6]) * lbmodel.w[2];
  }

  if (next[3] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 3] = (m[0] + m[2] - m[5] + m[6]) * lbmodel.w[3];
  }

  if (next[4] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 4] = (m[0] - m[2] - m[5] + m[6]) * lbmodel.w[4];
  }

  if (next[5] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 5] = (m[0] + m[3] - 2.*m[6]) * lbmodel.w[5];
  }

  if (next[6] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 6] = (m[0] - m[3] - 2.*m[6]) * lbmodel.w[6];
  }

  if (next[7] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 7] = (m[0] + m[1] + m[2] + m[4] + 2.*m[6] + m[7]) * lbmodel.w[7];
  }

  if (next[8] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 8] = (m[0] - m[1] - m[2] + m[4] + 2.*m[6] + m[7]) * lbmodel.w[8];
  }

  if (next[9] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][ 9] = (m[0] + m[1] - m[2] + m[4] + 2.*m[6] - m[7]) * lbmodel.w[9];
  }

  if (next[10] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][10] = (m[0] - m[1] + m[2] + m[4] + 2.*m[6] - m[7]) * lbmodel.w[10];
  }

  if (next[11] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][11] = (m[0] + m[1] + m[3] + m[4] + m[5] - m[6] + m[8]) * lbmodel.w[11];
  }

  if (next[12] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][12] = (m[0] - m[1] - m[3] + m[4] + m[5] - m[6] + m[8]) * lbmodel.w[12];
  }

  if (next[13] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][13] = (m[0] + m[1] - m[3] + m[4] + m[5] - m[6] - m[8]) * lbmodel.w[13];
  }

  if (next[14] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][14] = (m[0] - m[1] + m[3] + m[4] + m[5] - m[6] - m[8]) * lbmodel.w[14];
  }

  if (next[15] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][15] = (m[0] + m[2] + m[3] + m[4] - m[5] - m[6] + m[9]) * lbmodel.w[15];
  }

  if (next[16] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][16] = (m[0] - m[2] - m[3] + m[4] - m[5] - m[6] + m[9]) * lbmodel.w[16];
  }

  if (next[17] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][17] = (m[0] + m[2] - m[3] + m[4] - m[5] - m[6] - m[9]) * lbmodel.w[17];
  }

  if (next[18] != -1) {
    tree = (p8est_tree_t *) sc_array_index_int(p8est->trees, lbadapt_mesh->quad_to_tree[qid]);
    cell = (p8est_quadrant_t *) sc_array_index_int(&tree->quadrants, next[1]);
    data = (lbadapt_payload_t *) cell->p.user_data;
    data->lbfluid[1][18] = (m[0] - m[2] + m[3] + m[4] - m[5] - m[6] - m[9]) * lbmodel.w[18];
  }
#endif // !OLD_FLUCT

#else // D3Q19
  double **e = lbmodel.e;
  index_t next[lbmodel.n_veloc];
  for (int i = 0; i < lbmodel.n_veloc; i++) {
    next[i] = get_linear_index(c[i][0],c[i][1],c[i][2],lblattic.halo_grid);
    lbfluid[1][i][next[i]] = 0.0;
    for (int j = 0; j < lbmodel.n_veloc; j++)
      lbfluid[1][i][next[i]] += mode[j]*e[j][i]/e[19][j];
    lbfluid[1][i][index] *= w[i];
  }
#endif // D3Q19

  return 0;
}

/*** ITERATOR CALLBACKS ***/
void lbadapt_get_boundary_status (p8est_iter_volume_info_t * info, void * user_data) {
  p8est_t * p8est = info->p4est;                                /* get p8est */
  p8est_quadrant_t * q = info->quad;                            /* get current global cell id */
  p4est_topidx_t which_tree = info->treeid;                     /* get current tree id */
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
  p8est_quadrant_t * q = info->quad;                            /* get current global cell id */
  lbadapt_payload_t *data = (lbadapt_payload_t *) q->p.user_data; /* payload of cell */

  double h;                                                             /* local meshwidth */
  h = (double) P8EST_QUADRANT_LEN(q->level) / (double) P8EST_ROOT_LEN;

  // convert rho to lattice units
  double rho   = lbpar.rho[0] * h * h * h;
  // start with fluid at rest and no stress
  double j[3]  = {0., 0., 0.};
  double pi[6] = {0., 0., 0., 0., 0., 0.};
  lbadapt_calc_n_from_rho_j_pi (data->lbfluid, rho, j, pi, h);
}


void lbadapt_calc_local_rho (p8est_iter_volume_info_t * info, void * user_data) {
  double *rho = (double *) user_data;                           /* passed double to fill */
  p8est_quadrant_t * q = info->quad;                            /* get current global cell id */
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
  momentum[0] += j[0] + data->lbfields.force[0];
  momentum[1] += j[1] + data->lbfields.force[1];
  momentum[2] += j[2] + data->lbfields.force[2];

  momentum[0] *= h/lbpar.tau;
  momentum[1] *= h/lbpar.tau;
  momentum[2] *= h/lbpar.tau;
}


void lbadapt_calc_local_pi (p8est_iter_volume_info_t * info, void *user_data) {
  double *bnd_vals = (double *) user_data;                      /* passed array to fill */
  p8est_quadrant_t * q = info->quad;                            /* get current global cell id */
  p4est_topidx_t which_tree = info->treeid;                     /* get current tree id */
  p4est_locidx_t local_id = info->quadid;                       /* get cell id w.r.t. tree-id */
  p8est_tree_t * tree;
  lbadapt_payload_t *data = (lbadapt_payload_t *) q->p.user_data; /* payload of cell */

  double bnd;                                                     /* local meshwidth */
  p4est_locidx_t  arrayoffset;

  tree = p8est_tree_array_index (p8est->trees, which_tree);
  local_id += tree->quadrants_offset;   /* now the id is relative to the MPI process */
  arrayoffset = local_id;      /* each local quadrant has 2^d (P4EST_CHILDREN) values in u_interp */

  /* just grab the u value of each cell and pass it into solution vector */
  bnd = data->boundary;
  bnd_vals[arrayoffset] = bnd;
}


void lbadapt_collide_stream (p8est_iter_volume_info_t * info, void * user_data) {
  /* collect some dates from iteration info */
  p8est_quadrant_t * q = info->quad;
  lbadapt_payload_t *data = (lbadapt_payload_t *) q->p.user_data;
  double h;                                                     /* local meshwidth */
  h = (double) P8EST_QUADRANT_LEN(q->level) / (double) P8EST_ROOT_LEN;

  /* place for storing modes */
  double modes[19];

  /* calculate modes locally */
  lbadapt_calc_modes(data->lbfluid, modes);

  /* deterministic collisions */
  lbadapt_relax_modes(modes, data->lbfields.force, h);

  /* fluctuating hydrodynamics */

  if (fluct) lbadapt_thermalize_modes(modes, h);

  /* apply forces */
#ifdef EXTERNAL_FORCES
  lbadapt_apply_forces(modes, &data->lbfields, h);
#else // EXTERNAL_FORCES
  if (has_force) lbadapt_apply_forces(modes, &data->lbfields, h);
#endif // EXTERNAL_FORCES

  /* transform back to populations and streaming */
  lbadapt_calc_n_from_modes_push(info->quadid, modes);
}


void lbadapt_bounce_back (p8est_iter_volume_info_t * info, void * user_data) {
}


void lbadapt_swap_pointers (p8est_iter_volume_info_t * info, void * user_data) {
}
#endif // LB_ADAPTIVE
