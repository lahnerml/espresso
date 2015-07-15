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
#include <p8est_mesh.h>
#include <p8est_nodes.h>
#include <p8est_vtk.h>

#include "utils.hpp"
#include "lb.hpp"

/* "global variables" */
extern p8est_t              *p8est;
extern p8est_connectivity_t *conn;
extern p8est_ghost_t        *lbadapt_ghost;
extern p8est_mesh_t         *lbadapt_mesh;
extern lbadapt_payload_t    *lbadapt_ghost_data;

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

/** setup function
 */
void lbadapt_init (p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant);


/*** REFINEMENT ***/
/** Refinement function that refines all cells
 *
 * \param [in] p8est       The forest.
 * \param [in] which_tree  The tree in the forest containing \a q.
 * \param [in] quadrant    The Quadrant.
 */
int refine_uniform (p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant);


/** Refinement function that refines all cells with probability 0. with probability 0.55
 *
 * \param [in] p8est       The forest.
 * \param [in] which_tree  The tree in the forest containing \a q.
 * \param [in] quadrant    The Quadrant.
 */
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
 * \param [in]      rho        The fluids density.
 * \param [in]      j          The fluids velocity.
 * \param [in]      pi         The fluids stress tensor.
 * \param [in]      h          The meshwidth of the current cell
 */
int lbadapt_calc_n_from_rho_j_pi (double ** datafield,
                                  double rho,
                                  double * j,
                                  double * pi,
                                  double h);


/** Calculate modes for MRT scheme
 *
 * \param [in]      populations The population vector.
 * \param     [out] mode        The resulting modes to be relaxed in a later step.
 */
int lbadapt_calc_modes (double ** populations, double * mode);


/** Perform MRT Relaxation step
 *
 * \param [in][out] mode  kinematic modes of the fluid.
 * \param [in]      force Force that is applied on the fluid.
 * \param [in]      h     Meshwidth of current cell
 */
int lbadapt_relax_modes (double * mode, double * force, double h);


/** Thermalize modes
 *
 * \param [in][out] mode  The modes to be thermalized.
 * \param [in]      h     The local mesh width
 */
int lbadapt_relax_modes (double * mode, double h);


/** Apply force on fluid.
 *
 * \param [in][out] mode  The modes that the force is applied on.
 * \param [in]      force The force that is applied.
 * \param [in]      h     The local mesh width.
 */
int lbadapt_apply_forces (double * mode, LB_FluidNode * lbfields, double h);


/** Transfer modes back to populations
 *
 * \param     [out] populations  The resulting particle densities.
 * \param [in]      mode         The modes.
 */
int lbadapt_calc_pop_from_modes (double * populations, double * m);


/*** ITERATION CALLBACKS ***/
void lbadapt_get_boundary_status (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_get_boundary_values (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_init_force_per_cell (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_init_fluid_per_cell (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_calc_local_rho (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_calc_local_j (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_calc_local_pi (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_collide_stream (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_bounce_back (p8est_iter_volume_info_t * info, void * user_data);


void lbadapt_swap_pointers (p8est_iter_volume_info_t * info, void * user_data);
#endif //LB_ADAPTIVE_H
