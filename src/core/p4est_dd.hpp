#ifndef _P4EST_DD_H
#define _P4EST_DD_H

#include "cells.hpp"

#ifdef DD_P4EST
#include <p4est_to_p8est.h>
#include <p8est.h>

#define P4EST_DD_GUARD(call) call
#else
#define P4EST_DD_GUARD(call) do { \
  fprintf(stderr, "Error: P4est cellsystem is not compiled into this ESPResSo.\n"); \
  errexit(); \
} while(0)
#endif

#ifdef DD_P4EST
/** Get pointer to short-range MD p4est */
p4est_t* dd_p4est_get_p4est();
#endif

/** Get number of trees in current p4est_connectivity structure for given
 * direction.
 *
 * @param dir   Direction index, ordered as x, y, z.
 *              Defaults to x if not specified.
 * @return      Number of trees in specified direction
 */
int dd_p4est_get_n_trees(int dir);

int dd_p4est_full_shell_neigh(int cell, int neighidx);

// Check all particles if they have left their cell and move the to the right neighboring cell
void dd_p4est_exchange_and_sort_particles(int global_flag);

// Handles everything that needs to be done on a geometry change
void dd_p4est_on_geometry_change(int flags);

/** Initialize the topology. The argument is a list of cell pointers,
    containing particles that have to be sorted into new cells. The
    particles might not belong to this node.  This procedure is used
    when particle data or cell structure has changed and the cell
    structure has to be reinitialized. This also includes setting up
    the cell_structure array.
    @param cl List of cell pointers with particles to be stored in the
    new cell system.
*/
void dd_p4est_topology_init(CellPList *cl, bool isRepart = false);

/** Called when the current cell structure is invalidated because for
    example the box length has changed. This procedure may NOT destroy
    the old inner and ghost cells, but it should free all other
    organizational data. Note that parameters like the box length or
    the node_grid may already have changed. Therefore organizational
    data has to be stored independently from variables that may be
    changed from outside. */
// Same as dd_topology_release.
//void dd_p4est_topology_release();

void dd_p4est_write_parallel_vtk(char* filename);

/** Generate vtk files for particles.
 *
 * @param filename     Filename for output-files
 */
void dd_p4est_write_particle_vtk(char* filename);

/** Preprocessing before repart.
 * 
 * Enables the use of optimized resorting routine.
 */
void p4est_dd_repart_preprocessing();

#ifdef DD_P4EST
void p4est_dd_repartition(const std::string& desc, bool verbose);
#endif

#endif
