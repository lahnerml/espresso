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

// Creates the irregular DD using p4est
void dd_p4est_create_grid (bool isRepart = false);
// Compute communication partners for this node and fill internal lists
void dd_p4est_comm ();
// Mark all cells either local or ghost. Local cells are arranged before ghost cells
void dd_p4est_mark_cells ();
// Prepare a GhostCommunicator using the internal lists
void dd_p4est_prepare_comm (GhostCommunicator *comm, int data_part);

// Update the box length for a given communicator
void dd_p4est_update_comm_w_boxl(GhostCommunicator *comm);
// Fill the IA_NeighborList and compute the half-shell for p4est based DD
void dd_p4est_init_cell_interaction();

// Map a position to a cell, returns NULL if not in local (+ ROUND_ERR_PREC*boxl) domain
Cell* dd_p4est_save_position_to_cell(double pos[3]);
// Map a position to a cell, returns NULL if not in local domain
Cell* dd_p4est_position_to_cell(double pos[3]);
//void dd_p4est_position_to_cell(double pos[3], int* idx);
Cell* dd_p4est_position_to_cell_strict(double pos[3]);

// Check all particles if they have left their cell and move the to the right neighboring cell
void dd_p4est_exchange_and_sort_particles();

// Send all particles in cp to the right process
void dd_p4est_global_exchange_part(ParticleList* cp);

// Map a position to a global processor index
int dd_p4est_pos_to_proc(double pos[3]);
// Compute a Morton index for a position (this is not equal to the p4est index)
int64_t dd_p4est_pos_morton_idx(double pos[3]);

// Handles everything that needs to be done on a geometry change
void dd_p4est_on_geometry_change(int flags);

// Writes all local particles in a parallel VTK-particle file
void dd_p4est_write_particle_vtk(char *filename);

// Writes the MD grid as VTK file
void dd_p4est_write_vtk();

#ifdef DD_P4EST
// Repartition a given p4est along the MD grid, so that processor domain boundaries are aligned
void dd_p4est_partition(p4est_t *p4est, p4est_mesh_t *mesh, p4est_connectivity_t *conn);
#endif

void p4est_dd_repartition(const std::string& desc, bool debug);

#endif
