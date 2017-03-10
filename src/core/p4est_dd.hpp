#ifndef _P4EST_DD_H
#define _P4EST_DD_H

#include "utils.hpp"
#include "cells.hpp"

#ifndef DD_P4EST
#define P4EST_NOCHANGE
#endif
//#define P4EST_NOCHANGE

void dd_p4est_free ();
void dd_p4est_create_grid ();
void dd_p4est_comm ();
void dd_p4est_mark_cells ();
void dd_p4est_prepare_comm (GhostCommunicator *comm, int data_part);

void dd_p4est_update_comm_w_boxl(GhostCommunicator *comm);
void dd_p4est_init_cell_interaction();

Cell* dd_p4est_save_position_to_cell(double pos[3]);
Cell* dd_p4est_position_to_cell(double pos[3]);
//void dd_p4est_position_to_cell(double pos[3], int* idx);

void dd_p4est_exchange_and_sort_particles();

void dd_p4est_global_exchange_part(ParticleList* cp);

int dd_p4est_pos_to_proc(double pos[3]);
int64_t dd_p4est_pos_morton_idx(double pos[3]);
int64_t dd_p4est_cell_morton_idx(int x, int y, int z);

void dd_p4est_on_geometry_change(int flags);

void dd_p4est_write_particle_vtk(char *filename);

#endif
