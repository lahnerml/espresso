/*
  Copyright (C) 2010,2011,2012,2013,2014,2015 The ESPResSo project
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
/** \file lb-adaptive.hpp
 *
 * Adaptive Lattice Boltzmann Scheme using CPU.
 * Header file for \ref lb-adaptive.cpp.
 *
 */

#ifndef LB_ADAPTIVE_H
#define LB_ADAPTIVE_H

/* p4est includes; opted to go for pure 3D */
#include <p8est_connectivity.h>
#include <p8est_extended.h>
#include <p8est_ghost.h>
#include <p8est_iterate.h>
#include <p8est_nodes.h>
#include <p8est_vtk.h>

#include "utils.hpp"

/* "global variables" */
extern p8est_t              *p8est;
extern p8est_connectivity_t *conn;


/** setup function 
 */
void lbadapt_init (p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant);

/*** REFINEMENT ***/
int refine_uniform (p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant);


int refine_random (p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant);

/*** HELPER FUNCTIONS ***/
/* Geometry */
/** Get the coordinates of the midpoint of a quadrant.
 *
 * \param [in]  p4est    the forest
 * \param [in]  which_tree the tree in the forest containing \a q
 * \param [in]  q      the quadrant
 * \param [out] xyz    the coordinates of the midpoint of \a q
 */
void lbadapt_get_midpoint (p8est_t * p8est, p4est_topidx_t which_tree,
         p8est_quadrant_t * q, double xyz[3]);

/* LBM */
/** Calculate equilibrium distribution from given fluid parameters
 *
 * \param [in][out] datafield  The fluid node that shall be written.
 * \param [in]      rho        The fluid density.
 * \param [in]      j          The fluid velocity.
 * \param [in]      pi         The fluid's stress tensor.
 */
int lbadapt_calc_n_from_rho_j_pi (double * datafield, double rho, double* j, double* pi);

/*** ITERATION CALLBACKS ***/
void lbadapt_get_boundary_status (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_get_boundary_values (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_init_force_per_cell (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_init_fluid_per_cell (p8est_iter_volume_info_t * info, void * user_data);


#endif //LB_ADAPTIVE_H
