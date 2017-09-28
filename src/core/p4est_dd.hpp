#ifndef _P4EST_DD_H
#define _P4EST_DD_H

#include "utils.hpp"
#include "cells.hpp"

#ifdef DD_P4EST
#include <p4est_to_p8est.h>
#include <p8est_mesh.h>
#endif

// P4EST_NOCHANGE is used for debugging
#ifndef DD_P4EST
#define P4EST_NOCHANGE 
#endif
//#define P4EST_NOCHANGE
int dd_p4est_full_shell_neigh(int cell, int neighidx);

// Free the memory and all p4est related stuff
void dd_p4est_free ();

// Creates the irregular DD using p4est
void dd_p4est_create_grid ();
// Compute communication partners for this node and fill internal lists
void dd_p4est_comm ();
// Mark all cells either local or ghost. Local cells are arranged before ghost cells
void dd_p4est_mark_cells ();
// Prepare a GhostCommunicator using the internal lists
void dd_p4est_prepare_comm (GhostCommunicator *comm, int data_part);

// Update the box length for a given communicator
void dd_p4est_update_comm_w_boxl(GhostCommunicator *comm);
// Fill the IA_NeighborList and compute the half-shell for p4est based DD
void dd_p4est_init_cell_interactions();

// Map a position to a cell, returns NULL if not in local (+ ROUND_ERR_PREC*boxl) domain
Cell* dd_p4est_save_position_to_cell(double pos[3]);
// Map a position to a cell, returns NULL if not in local domain
Cell* dd_p4est_position_to_cell(double pos[3]);
//void dd_p4est_position_to_cell(double pos[3], int* idx);

// Check all particles if they have left their cell and move the to the right neighboring cell
void dd_p4est_exchange_and_sort_particles(int global_flag);

// Map a position to a global processor index
int dd_p4est_pos_to_proc(double pos[3]);
// Compute a Morton index for a position (this is not equal to the p4est index)
int64_t dd_p4est_pos_morton_idx(double pos[3]);

// Handles everything that needs to be done on a geometry change
void dd_p4est_on_geometry_change(int flags);

// Writes all local particles in a parallel VTK-particle file
void dd_p4est_write_particle_vtk(char *filename);

void dd_p4est_position_to_cell_indices(double pos[3],int* idx);

/** Revert the order of a communicator: After calling this the
    communicator is working in reverted order with exchanged
    communication types GHOST_SEND <-> GHOST_RECV. */
void dd_p4est_revert_comm_order (GhostCommunicator *comm);

/** Initialize the topology. The argument is a list of cell pointers,
    containing particles that have to be sorted into new cells. The
    particles might not belong to this node.  This procedure is used
    when particle data or cell structure has changed and the cell
    structure has to be reinitialized. This also includes setting up
    the cell_structure array.
    @param cl List of cell pointers with particles to be stored in the
    new cell system.
*/
void dd_p4est_topology_init(CellPList *cl);

/** Called when the current cell structure is invalidated because for
    example the box length has changed. This procedure may NOT destroy
    the old inner and ghost cells, but it should free all other
    organizational data. Note that parameters like the box length or
    the node_grid may already have changed. Therefore organizational
    data has to be stored independently from variables that may be
    changed from outside. */
void dd_p4est_topology_release();

void dd_p4est_update_communicators_w_boxl();

// Writes the MD grid as VTK file
void dd_p4est_write_vtk();

#ifdef DD_P4EST
// Repartition a given p4est along the MD grid, so that processor domain boundaries are aligned
void dd_p4est_partition(p4est_t *p4est, p4est_mesh_t *mesh, p4est_connectivity_t *conn);
#endif

void p4est_dd_repartition(const std::string& desc, bool debug);

#endif
