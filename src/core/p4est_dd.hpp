#ifndef _P4EST_DD_H
#define _P4EST_DD_H

#include "utils.hpp"
#include "cells.hpp"

#ifdef DD_P4EST
#include <p4est_to_p8est.h>
#include <p8est_mesh.h>
#endif

//#define P4EST_NOCHANGE
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

#ifdef DD_P4EST
// Repartition a given p4est along the MD grid, so that processor domain boundaries are aligned
void dd_p4est_partition(p4est_t *p4est, p4est_mesh_t *mesh, p4est_connectivity_t *conn);
#endif

void p4est_dd_repartition(const std::string& desc, bool debug);

double p4est_dd_imbalance(const std::string& desc);

#endif
