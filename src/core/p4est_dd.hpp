#ifndef _P4EST_DD_H
#define _P4EST_DD_H

#include "utils.hpp"
#include "cells.hpp"

#define P4EST_NOCHANGE

void dd_p4est_free ();
void dd_p4est_create_grid ();
void dd_p4est_comm ();
void dd_p4est_mark_cells ();
void dd_p4est_prepare_comm (GhostCommunicator *comm, int data_part);

// TODO
void dd_p4est_update_comm_w_boxl();
void dd_p4est_init_cell_interaction();

Cell* dd_p4est_save_position_to_cell(double pos[3]);
Cell* dd_p4est_position_to_cell(double pos[3]);
void dd_p4est_position_to_cell(double pos[3], int* idx);

void dd_p4est_exchange_and_sort_particles(int global_flag);

// Minors
/// dd_topology_release
/// dd_topology_init
/// calc_processor_min_num_cells

#endif
