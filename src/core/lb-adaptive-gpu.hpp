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
 * Header file for \ref lb-adaptive-gpu.cpp.
 *
 */

#ifndef _LB_ADAPTIVE_GPU_H
#define _LB_ADAPTIVE_GPU_H

#ifdef LB_ADAPTIVE_GPU
#include <p8est_vtk.h>

/** CAUTION: CUDA does not work along with Boost::MPI.
 *           Therefore it is not possible to just include lb.hpp and be fine.
 *           Instead we have to redefine a multitude of things which is more
 *           than ugly.
 */

/** \name Parameter fields for Lattice Boltzmann
 * The numbers are referenced in \ref mpi_bcast_lb_params
 * to determine what actions have to take place upon change
 * of the respective parameter. */
/*@{*/
#define LBPAR_DENSITY   0 /**< fluid density */
#define LBPAR_VISCOSITY 1 /**< fluid kinematic viscosity */
#define LBPAR_AGRID     2 /**< grid constant for fluid lattice */
#define LBPAR_TAU       3 /**< time step for fluid propagation */
#define LBPAR_FRICTION  4 /**< friction coefficient for viscous coupling between particles and fluid */
#define LBPAR_EXTFORCE  5 /**< external force acting on the fluid */
#define LBPAR_BULKVISC  6 /**< fluid bulk viscosity */

/** Note these are used for binary logic so should be powers of 2 */
#define LB_COUPLE_NULL        1
#define LB_COUPLE_TWO_POINT   2
#define LB_COUPLE_THREE_POINT 4

#define LBADAPT_PATCHSIZE 8
#define LBADAPT_PATCHSIZE_HALO 2 + LBADAPT_PATCHSIZE

#define P4EST_VTK_CELL_TYPE 11 /* VTK_VOXEL */
#define P4EST_ENABLE_VTK_BINARY 1
#define P4EST_ENABLE_VTK_COMPRESSION 1

typedef float lb_float;

/* temporary test*/
typedef struct {
  lb_float thread_idx[LBADAPT_PATCHSIZE_HALO][LBADAPT_PATCHSIZE_HALO]
                     [LBADAPT_PATCHSIZE_HALO];
  lb_float block_idx[LBADAPT_PATCHSIZE_HALO][LBADAPT_PATCHSIZE_HALO]
                    [LBADAPT_PATCHSIZE_HALO];
} test_grid_t;

typedef struct {
  lb_float rho[LB_COMPONENTS];
  lb_float viscosity[LB_COMPONENTS];
  lb_float bulk_viscosity[LB_COMPONENTS];
  lb_float agrid;
  lb_float tau;
  
  /** the initial level based on which the number of LB steps is defined */
  int base_level;
  /** the maximum refinement level */
  int max_refinement_level;
  
  lb_float friction[LB_COMPONENTS];
  lb_float ext_force[3]; /* Open question: Do we want a local force or global force? */
  lb_float rho_lb_units[LB_COMPONENTS];
  lb_float gamma_odd[LB_COMPONENTS];
  lb_float gamma_even[LB_COMPONENTS];
  bool is_TRT;
  int resend_halo;
} LB_Parameters;

typedef struct {
  int n_veloc ;
  lb_float (*c)[3];
  lb_float (*coeff)[4];
  lb_float (*w);
  lb_float **e;
  lb_float c_sound_sq;
} LB_Model;

typedef struct lbadapt_patch_cell {
  lb_float lbfluid[2][19];
  lb_float modes[19];
  int boundary;
  lb_float force[3];
} lbadapt_patch_cell_t;

typedef struct lbadapt_payload {
  int boundary;
  lbadapt_patch_cell_t patch[LBADAPT_PATCHSIZE_HALO][LBADAPT_PATCHSIZE_HALO]
                            [LBADAPT_PATCHSIZE_HALO];
} lbadapt_payload_t;

/** Opaque context type for writing VTK output with multiple function calls.
 */
typedef struct lbadapt_vtk_context {
  /* data passed initially */
  char *filename; /**< Original filename provided is copied. */
  char vtk_float_name[8];
  
  /* internal context data */
  int writing;                    /**< True after p4est_vtk_write_header. */
  int num_corners;     /**< Number of local element corners. */
  int num_points;      /**< Number of VTK points written. */
  int *node_to_corner; /**< Map a node to an element corner. */
  char vtufilename[BUFSIZ];       /**< Each process writes one. */
  char pvtufilename[BUFSIZ];      /**< Only root writes this one. */
  char visitfilename[BUFSIZ];     /**< Only root writes this one. */
  FILE *vtufile;                  /**< File pointer for the VTU file. */
  FILE *pvtufile;                 /**< Paraview meta file. */
  FILE *visitfile;                /**< Visit meta file. */
} lbadapt_vtk_context_t;

extern int local_num_quadrants;

extern LB_Parameters lbpar;
extern LB_Model lbmodel;

/* int to indicate fluctuations */
extern int fluct;

/** Switch indicating momentum exchange between particles and fluid */
extern int transfer_momentum;

/** Eigenvalue of collision operator corresponding to shear viscosity. */
extern double lblambda;

/** Eigenvalue of collision operator corresponding to bulk viscosity. */
extern double lblambda_bulk;

extern double prefactors[P8EST_MAXLEVEL];

extern double gamma_shear[P8EST_MAXLEVEL];

extern double gamma_bulk[P8EST_MAXLEVEL];

extern double gamma_odd;
extern double gamma_even;
extern double lb_phi[19];
extern double lb_coupl_pref;
extern double lb_coupl_pref2;

void test (test_grid_t *data_host);

/** This is actually taken from p4est and extended to patches. :P */
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
 * \param [in] write_qid   Boolean to determine if the quadrant id should be
 *                         written to vtk file.
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

#endif // _LB_ADAPTIVE_GPU_H
