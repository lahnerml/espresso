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
 * Adaptive Lattice Boltzmann Scheme.
 * Header file for \ref lb-adaptive.cpp.
 *
 */

#ifndef _LB_ADAPTIVE_H
#define _LB_ADAPTIVE_H

#ifdef LB_ADAPTIVE
/* p4est includes; opted to go for pure 3D */
#include <p8est_connectivity.h>
#include <p8est_extended.h>
#include <p8est_ghost.h>
#include <p8est_ghostvirt.h>
#include <p8est_iterate.h>
#include <p8est_mesh.h>
#include <p8est_meshiter.h>
#include <p8est_nodes.h>
#include <p8est_vtk.h>

#include "lb.hpp"
#include "utils.hpp"

/* "global variables" */
extern p8est_t *p8est;
extern p8est_connectivity_t *conn;
extern p8est_ghost_t *lbadapt_ghost;
extern p8est_ghostvirt_t *lbadapt_ghost_virt;
extern p8est_mesh_t *lbadapt_mesh;
extern lbadapt_payload_t **lbadapt_local_data;
extern lbadapt_payload_t **lbadapt_ghost_data;
extern int coarsest_level_local;
extern int finest_level_local;
extern int coarsest_level_ghost;
extern int finest_level_ghost;
extern int finest_level_global;

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

/** setup function for lbadapt_payload_t by setting as many values as possible
 *  to 0
 */
void lbadapt_init();

/** Init cell-local force values
 */
void lbadapt_reinit_force_per_cell();

/** (Re-)initialize the fluid according to the given value of rho
 */
void lbadapt_reinit_fluid_per_cell();

/** Get maxlevel of p4est
 */
int lbadapt_get_global_maxlevel();

#ifdef LB_ADAPTIVE_GPU
/** Populate the halos of patches of a specific level
 */
void lbadapt_patches_populate_halos(int level);
#endif // LB_ADAPTIVE_GPU

/** interpolating function
 *
 * \param [in]      p8est        The forest
 * \param [in]      which_tree   The tree in the forest \a q.
 * \param [in]      num_outgoing The number of quadrants being replaced.
 *                               1 for refinement, 8 for coarsening.
 * \param [in]      outgoing     The actual quadrants that will be replaced.
 * \param [in]      num_incoming The number of quadarants that will be added.
 * \param [in][out] incoming     Quadrants whose data needs to be initialized.
 */
void lbadapt_replace_quads(p8est_t *p8est, p4est_topidx_t which_tree,
                           int num_outgoing, p8est_quadrant_t *outgoing[],
                           int num_incoming, p8est_quadrant_t *incoming[]);

/*** LOAD BALANCING ***/
/** Weighting function for p4est_partition
 *
 * \param [in] p8est       The forest.
 * \param [in] which_tree  The tree in the forest containing \a q.
 * \param [in] quadrant    The Quadrant.
 * @returns quadrants weight according to subcycling
 */
int lbadapt_partition_weight(p8est_t *p8est, p4est_topidx_t which_tree,
                             p8est_quadrant_t *q);
/*** REFINEMENT ***/
/** Refinement function that refines all cells
 *
 * \param [in] p8est       The forest.
 * \param [in] which_tree  The tree in the forest containing \a q.
 * \param [in] quadrant    The Quadrant.
 */
int refine_uniform(p8est_t *p8est, p4est_topidx_t which_tree,
                   p8est_quadrant_t *quadrant);

/** Refinement function that refines all cells with probability 0.55
 *
 * \param [in] p8est       The forest.
 * \param [in] which_tree  The tree in the forest containing \a q.
 * \param [in] quadrant    The Quadrant.
 */
int refine_random(p8est_t *p8est, p4est_topidx_t which_tree,
                  p8est_quadrant_t *quadrant);

/** Refinement function that refines all cells for whichs anchor point holds
 * 0.25 <= z < 0.75
 *
 * \param [in] p8est       The forest.
 * \param [in] which_tree  The tree in the forest containing \a q.
 * \param [in] quadrant    The Quadrant.
 */
int refine_regional(p8est_t *p8est, p4est_topidx_t which_tree,
                    p8est_quadrant_t *q);

/** Refinement function that refines all cells whose midpoint is closer to a
 * boundary than half the cells side length.
 *
 * \param [in] p8est       The forest.
 * \param [in] which_tree  The tree in the forest containing \a q.
 * \param [in] quadrant    The Quadrant.
 */
int refine_geometric(p8est_t *p8est, p4est_topidx_t which_tree,
                     p8est_quadrant_t *q);

/*** HELPER FUNCTIONS ***/
/* Geometry */
/** Get the coordinates of the midpoint of a quadrant.
 *
 * \param [in]  p8est    the forest
 * \param [in]  which_tree the tree in the forest containing \a q
 * \param [in]  q      the quadrant
 * \param [out] xyz    the coordinates of the midpoint of \a q
 */
void lbadapt_get_midpoint(p8est_t *p8est, p4est_topidx_t which_tree,
                          p8est_quadrant_t *q, lb_float xyz[3]);

/** Get the coordinates of the midpoint of a quadrant
 *
 * \param [in]  mesh_iter  A mesh-based iterator.
 * \param [out] xyz        The coordinates of the the midpoint of the current
 *                         quadrant that mesh_iter is pointing to.
 */
void lbadapt_get_midpoint(p8est_meshiter_t *mesh_iter, lb_float xyz[3]);

/** Get the coordinates of the front lower left corner of a quadrant.
 *
 * \param [in]  p8est    the forest
 * \param [in]  which_tree the tree in the forest containing \a q
 * \param [in]  q      the quadrant
 * \param [out] xyz    the coordinates of the midpoint of \a q
 */
void lbadapt_get_front_lower_left(p8est_t *p8est, p4est_topidx_t which_tree,
                                  p8est_quadrant_t *q, lb_float xyz[3]);

/** Get the coordinates of the front lower left corner of a quadrant
 *
 * \param [in]  mesh_iter  A mesh-based iterator.
 * \param [out] xyz        The coordinates of the the front lower left corner
 *                         of the current quadrant that mesh_iter is pointing
 *                         to.
 */
void lbadapt_get_front_lower_left(p8est_meshiter_t *mesh_iter, lb_float xyz[3]);

/* LBM */
/** Calculate equilibrium distribution from given fluid parameters
 *
 * \param [in][out] datafield  The fluid node that shall be written.
 * \param [in]      rho        The fluids density.
 * \param [in]      j          The fluids velocity.
 * \param [in]      pi         The fluids stress tensor.
 * \param [in]      h          The local mesh-width.
 */
int lbadapt_calc_n_from_rho_j_pi(lb_float datafield[2][19], lb_float rho,
                                 lb_float *j, lb_float *pi, lb_float h);

/** Calculate modes for MRT scheme
 *
 * \param [in]      populations The population vector.
 * \param     [out] mode        The resulting modes to be relaxed in a later
 * step.
 */
int lbadapt_calc_modes(lb_float populations[2][19], lb_float *mode);

/** Perform MRT Relaxation step
 *
 * \param [in][out] mode  kinematic modes of the fluid.
 * \param [in]      force Force that is applied on the fluid.
 * \param [in]      h     Meshwidth of current cell
 */
int lbadapt_relax_modes(lb_float *mode, lb_float *force, lb_float h);

/** Thermalize modes
 *
 * \param [in][out] mode  The modes to be thermalized.
 */
int lbadapt_relax_modes(lb_float *mode);

/** Apply force on fluid.
 *
 * \param [in][out] mode  The modes that the force is applied on.
 * \param [in]      force The force that is applied.
 * \param [in]      h     The local mesh width.
 */
int lbadapt_apply_forces(lb_float *mode, LB_FluidNode *lbfields, lb_float h);

/** Transfer modes back to populations
 *
 * \param     [out] populations  The resulting particle densities.
 * \param [in]      m            The modes.
 */
int lbadapt_calc_pop_from_modes(lb_float *populations, lb_float *m);

/** collision
 * CAUTION: sync ghost data after collision
 *
 * \param [in] level   The level on which to perform the collision step
 */
void lbadapt_collide(int level);

/** Populate virtual cells with post-collision values from their respective
 * father cell
 *
 * \param [in] level   The level of the real cells whose virtual subcells are
 *                     populated.
 */
void lbadapt_populate_virtuals(int level);

/** streaming
 * CAUTION: sync ghost data before streaming
 *
 * \param [in] level   The level on which to perform the streaming step
 */
void lbadapt_stream(int level);

/** bounce back
 * CAUTION: sync ghost data before streaming
 *
 * \param [in] level   The level on which to perform the bounce-back step
 */
void lbadapt_bounce_back(int level);

/** Update population of real cells from streaming steps from neighboring
 * quadrants.
 *
 * \param [in] level   The level of the real cells whose populations are updated
 *                     from their respective virtual subcells.
 */
void lbadapt_update_populations_from_virtuals(int level);

/** swap pre- and poststreaming pointers
 *
 * \param [in] level   The level on which to swap lbfluid pointers
 */
void lbadapt_swap_pointers(int level);

/** Obtain boundary information of each quadrant for vtk output
 */
void lbadapt_get_boundary_values(sc_array_t *boundary_values);

/** Obtain density values of each quadrant for vtk output
 */
void lbadapt_get_density_values(sc_array_t *density_values);

/** Obtain velocity values of each quadrant for vtk output
 */
void lbadapt_get_velocity_values(sc_array_t *velocity_values);

/** Init boudary flag of each quadrant.
 */
void lbadapt_get_boundary_status();

/** Calculate local density from pre-collision moments
 *
 * \param [in]  mesh_iter    mesh-based iterator
 * \param [out] rho          density
 */
void lbadapt_calc_local_rho(p8est_meshiter_t *mesh_iter, lb_float *rho);

/** Calculate local fluid velocity from pre-collision moments
 *
 * \param [in]  mesh_iter    mesh-based iterator
 * \param [out] j            velocity
 */
void lbadapt_calc_local_j(p8est_meshiter_t *mesh_iter, lb_float *j);

/*** ITERATION CALLBACKS ***/
void lbadapt_set_recalc_fields(p8est_iter_volume_info_t *info, void *user_data);

void lbadapt_calc_local_rho(p8est_iter_volume_info_t *info, void *user_data);

void lbadapt_calc_local_j(p8est_iter_volume_info_t *info, void *user_data);

void lbadapt_calc_local_pi(p8est_iter_volume_info_t *info, void *user_data);

void lbadapt_dump2file(p8est_iter_volume_info_t *info, void *user_data);

#ifdef LB_ADAPTIVE_GPU
/** This is actually taken from p4est and extended to patches. :P */
#define P4EST_VTK_CELL_TYPE 11 /* VTK_VOXEL */
#define P4EST_ENABLE_VTK_BINARY 1
#define P4EST_ENABLE_VTK_COMPRESSION 1

/** Opaque context type for writing VTK output with multiple function calls.
 */
typedef struct lbadapt_vtk_context {
  /* data passed initially */
  char *filename; /**< Original filename provided is copied. */
  char vtk_float_name[8];

  /* internal context data */
  int writing;                    /**< True after p4est_vtk_write_header. */
  p4est_locidx_t num_corners;     /**< Number of local element corners. */
  p4est_locidx_t num_points;      /**< Number of VTK points written. */
  p4est_locidx_t *node_to_corner; /**< Map a node to an element corner. */
  p8est_nodes_t *nodes;           /**< NULL? depending on scale/continuous. */
  char vtufilename[BUFSIZ];       /**< Each process writes one. */
  char pvtufilename[BUFSIZ];      /**< Only root writes this one. */
  char visitfilename[BUFSIZ];     /**< Only root writes this one. */
  FILE *vtufile;                  /**< File pointer for the VTU file. */
  FILE *pvtufile;                 /**< Paraview meta file. */
  FILE *visitfile;                /**< Visit meta file. */
} lbadapt_vtk_context_t;

/** The first call to write a VTK file using individual functions.
 *
 * Writing a VTK file is split into multiple functions that keep a context.
 * This is the first function that allocates the opaque context structure.
 * After allocation, further parameters can be set for the context.
 * Then, the header, possible data fields, and the footer must be written.
 * The process can be aborted any time by destroying the context.  In this
 * case, open files are closed cleanly with only partially written content.
 *
 * \param p4est     The p8est to be written.
 *                  If no geometry is specified in
 *                  \ref p8est_vtk_context_set_geom, we require
 *                  \b p8est->connectivity to have valid vertex arrays.
 * \param filename  The first part of the name which will have the processor
 *                  number appended to it (i.e., the output file will be
 *                  filename_rank.vtu).  The parallel meta-files for Paraview
 *                  and Visit use this basename too.
 *                  We copy this filename to internal storage, so it is not
 *                  needed to remain alive after calling this function.
 * \return          A VTK context fur further use.
 */
lbadapt_vtk_context_t *lbadapt_vtk_context_new(const char *filename);

/** Cleanly destroy a \ref p8est_vtk_context_t structure.
 *
 * This function closes all the file pointers and frees the context.
 * Tt can be called even if the VTK output
 * has only been partially written, the files' content will be incomplete.
 *
 * \param[in] context     The VTK file context to be destroyed.
 */
void lbadapt_vtk_context_destroy(lbadapt_vtk_context_t *context);

/** Write the VTK header.
 *
 * Writing a VTK file is split into a few routines.
 * This allows there to be an arbitrary number of
 * fields.  The calling sequence would be something like
 *
 *     vtk_context = p8est_vtk_context_new (p8est, "output");
 *     p8est_vtk_context_set_* (vtk_context, parameter);
 *     vtk_context = p8est_vtk_write_header (vtk_context, ...);
 *     if (vtk_context == NULL) { error; }
 *     vtk_context = p8est_vtk_write_cell_data (vtk_context, ...);
 *     if (vtk_context == NULL) { error; }
 *     vtk_context = p8est_vtk_write_point_data (vtk_context, ...);
 *     if (vtk_context == NULL) { error; }
 *     retval = p8est_vtk_write_footer (vtk_context);
 *     if (retval) { error; }
 *
 * \param [in,out] cont    A VTK context created by \ref p8est_vtk_context_new.
 *                         None of the vtk_write functions must have been
 * called.
 *                         This context is the return value if no error occurs.
 *
 * \return          On success, an opaque context (p8est_vtk_context_t) pointer
 *                  that must be passed to subsequent p8est_vtk calls.  It is
 *                  required to call \ref p8est_vtk_write_footer eventually with
 *                  this value.  Returns NULL on error.
 */
lbadapt_vtk_context_t *lbadapt_vtk_write_header(lbadapt_vtk_context_t *cont);

/** Write VTK cell data.
 *
 * There are options to have this function write
 * the tree id, quadrant level, or MPI rank without explicit input data.
 *
 * Writing a VTK file is split into a few routines.
 * This allows there to be an arbitrary number of
 * fields.
 *
 * \param [in,out] cont    A VTK context created by \ref p8est_vtk_context_new.
 * \param [in] write_tree  Boolean to determine if the tree id should be output.
 * \param [in] write_level Boolean to determine if the tree levels should be
 * output.
 * \param [in] write_rank  Boolean to determine if the MPI rank should be
 * output.
 * \param [in] wrap_rank   Number to wrap around the rank with a modulo
 * operation.
 *                         Can be 0 for no wrapping.
 * \param [in] num_cell_scalars Number of cell scalar datasets to output.
 * \param [in] num_cell_vectors Number of cell vector datasets to output.
 *
 * The variable arguments need to be pairs of (fieldname, fieldvalues), followed
 * by a final argument of the VTK context cont (same as the first argument).
 * The cell scalar pairs come first, followed by the cell vector pairs, then
 * cont.
 * Each 'fieldname' argument shall be a char string containing the name of the
 * data
 * contained in the following 'fieldvalues'.  Each of the 'fieldvalues'
 * arguments shall be an sc_array_t * holding lb_float variables.  The number of
 * lb_floats in each sc_array must be exactly \a p4est->local_num_quadrants for
 * scalar data and \a 3*p4est->local_num_quadrants for vector data.
 *
 * \note The current p8est_vtk_context_t structure, \a cont, must be the first
 * and the last argument
 * of any call to this function; this argument is used to validate that the
 * correct number of variable arguments have been provided.
 *
 * \return          On success, the context that has been passed in.
 *                  On failure, returns NULL and deallocates the context.
 */
lbadapt_vtk_context_t *
lbadapt_vtk_write_cell_dataf(lbadapt_vtk_context_t *cont, int write_tree,
                             int write_level, int write_rank, int wrap_rank,
                             int write_qid, int num_cell_scalars,
                             int num_cell_vectors, ...);

/** Write the VTU footer and clean up.
 *
 * Writing a VTK file is split into a few routines.
 * This function writes the footer information to the VTK file and cleanly
 * destroys the VTK context.
 *
 * \param [in] cont Context is deallocated before the function returns.
 *
 * \return          This returns 0 if no error and -1 if there is an error.
 */
int lbadapt_vtk_write_footer(lbadapt_vtk_context_t *cont);
#endif // LB_ADAPTIVE_GPU

#endif // LB_ADAPTIVE

#endif // _LB_ADAPTIVE_H
