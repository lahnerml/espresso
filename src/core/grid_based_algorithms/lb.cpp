/*
  Copyright (C) 2010-2018 The ESPResSo project
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
/** \file
 *
 * Lattice Boltzmann algorithm for hydrodynamic degrees of freedom.
 *
 * Includes fluctuating LB and coupling to MD particles via frictional
 * momentum transfer.
 *
 */

#include "grid_based_algorithms/lb.hpp"
#include "grid_based_algorithms/lbgpu.hpp"
#include "nonbonded_interactions/nonbonded_interaction_data.hpp"
#include <cinttypes>

#ifdef LB

#include "cells.hpp"
#include "communication.hpp"
#include "global.hpp"
#include "grid.hpp"
#include "grid_based_algorithms/lb-adaptive.hpp"
#include "grid_based_algorithms/lbboundaries.hpp"
#include "halo.hpp"
#include "lb-d3q19.hpp"
#include "p4est_utils.hpp"
#include "thermostat.hpp"
#include "virtual_sites/lb_inertialess_tracers.hpp"

#include <boost/multi_array.hpp>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "cuda_interface.hpp"

#ifndef LB_ADAPTIVE_GPU
#ifdef ADDITIONAL_CHECKS
static void lb_check_halo_regions(const LB_Fluid &lbfluid);
void print_fluid();
#endif // ADDITIONAL_CHECKS

/** Flag indicating momentum exchange between particles and fluid */
int transfer_momentum = 0;

/** Struct holding the Lattice Boltzmann parameters */
// LB_Parameters lbpar = { .rho={0.0}, .viscosity={0.0}, .bulk_viscosity={-1.0},
// .agrid=-1.0, .tau=-1.0, .friction={0.0}, .ext_force_density={ 0.0, 0.0,
// 0.0},.rho_lb_units={0.},.gamma_odd={0.}, .gamma_even={0.} };
LB_Parameters lbpar = {
    // rho
    0.0,
    // viscosity
    0.0,
    // bulk_viscosity
    -1.0,
    // agrid
    -1.0,
    // tau
    -1.0,
    // friction
    0.0,
    // ext_force_density
    {0.0, 0.0, 0.0},
    // rho_lb_units
    0.,
    // gamma_odd
    0.,
    // gamma_even
    0.,
// gamma_shear
#ifdef LB_ADAPTIVE
    {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
#else
    0.,
#endif
// gamma_bulk
#ifdef LB_ADAPTIVE
    {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
#else
    0.,
#endif
    // is_TRT
    false,
    // resend_halo
    0,
    // fluct
    0,
    // phi
    {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0.}};

/** The DnQm model to be used. */
LB_Model<> lbmodel = {d3q19_lattice, d3q19_coefficients, d3q19_w,
                      d3q19_modebase, 1. / 3.};

#if (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) && !defined(GAUSSRANDOM))
#define FLATNOISE
#endif // (!defined(FLATNOISE) && !defined(GAUSSRANDOMCUT) &&
       // !defined(GAUSSRANDOM))

#ifndef LB_ADAPTIVE
/** The underlying lattice structure */
Lattice lblattice;

using LB_FluidData = boost::multi_array<double, 2>;
static LB_FluidData lbfluid_a;
static LB_FluidData lbfluid_b;

/** Pointer to the velocity populations of the fluid.
 * lbfluid contains pre-collision populations, lbfluid_post
 * contains post-collision */
LB_Fluid lbfluid;
LB_Fluid lbfluid_post;

/** Pointer to the hydrodynamic fields of the fluid nodes */
std::vector<LB_FluidNode> lbfields;

/** Communicator for halo exchange between processors */
HaloCommunicator update_halo_comm = {0, nullptr};
#else
int n_lbsteps = 0;
#endif // !LB_ADAPTIVE

/*@{*/

/** amplitude of the fluctuations in the viscous coupling */
static double lb_coupl_pref = 0.0;
/** amplitude of the fluctuations in the viscous coupling with Gaussian random
 * numbers */
static double lb_coupl_pref2 = 0.0;
/*@}*/

/** measures the MD time since the last fluid update */
static double fluidstep = 0.0;

#ifdef ADDITIONAL_CHECKS
/** counts the random numbers drawn for fluctuating LB and the coupling */
static int rancounter = 0;
#endif // ADDITIONAL_CHECKS

/***********************************************************************/
#endif // !LB_ADAPTIVE_GPU
#endif // LB

#if defined(LB) || defined(LB_GPU)

#include "errorhandling.hpp"
#include "global.hpp"
#include "grid.hpp"

/* *********************** C Interface part
 * *************************************/
/* ******************************************************************************/

/*
 * set lattice switch on C-level
 */
int lb_set_lattice_switch(int py_switch) {
  switch (py_switch) {
  case 0:
    lattice_switch = LATTICE_OFF;
    mpi_bcast_parameter(FIELD_LATTICE_SWITCH);
    return 0;
  case 1:
    lattice_switch = LATTICE_LB;
    mpi_bcast_parameter(FIELD_LATTICE_SWITCH);
    return 0;
  case 2:
    lattice_switch = LATTICE_LB_GPU;
    mpi_bcast_parameter(FIELD_LATTICE_SWITCH);
    return 0;
  default:
    return 1;
  }
}

/*
 * get lattice switch on py-level
 */
int lb_get_lattice_switch(int *py_switch) {
  if (lattice_switch) {
    *py_switch = lattice_switch;
    return 0;
  } else
    return 1;
}

#ifdef SHANCHEN
int lb_lbfluid_set_shanchen_coupling(double *p_coupling) {
#ifdef LB_GPU
  int ii, jj, n = 0;
  switch (LB_COMPONENTS) {
  case 1:
    lbpar_gpu.coupling[0] = (float)p_coupling[0];
    lbpar_gpu.coupling[1] = (float)p_coupling[1];
    break;
  default:
    for (ii = 0; ii < LB_COMPONENTS; ii++) {
      for (jj = ii; jj < LB_COMPONENTS; jj++) {
        lbpar_gpu.coupling[LB_COMPONENTS * ii + jj] = (float)p_coupling[n];
        lbpar_gpu.coupling[LB_COMPONENTS * jj + ii] = (float)p_coupling[n];
        n++;
      }
    }
    break;
  }
  on_lb_params_change_gpu(LBPAR_COUPLING);
#endif // LB_GPU
#ifdef LB
#error not implemented
#endif // LB
  return 0;
}

int lb_lbfluid_set_mobility(double *p_mobility) {
  int ii;
  for (ii = 0; ii < LB_COMPONENTS - 1; ii++) {
    if (p_mobility[ii] <= 0) {
      return -1;
    }
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      lbpar_gpu.mobility[ii] = (float)p_mobility[ii];
      on_lb_params_change_gpu(LBPAR_MOBILITY);
#endif // LB_GPU
    } else {
#ifdef LB
#error not implemented
#endif // LB
    }
  }
  return 0;
}

int affinity_set_params(int part_type_a, int part_type_b, double *affinity) {
  IA_parameters *data = get_ia_param_safe(part_type_a, part_type_b);
  data->affinity_on = 0;
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (affinity[ii] < 0 || affinity[ii] > 1) {
      return ES_ERROR;
    }
    data->affinity[ii] = affinity[ii];
    if (data->affinity[ii] > 0)
      data->affinity_on = 1;
  }

  /* broadcast interaction parameters */
  mpi_bcast_ia_params(part_type_a, part_type_b);

  return ES_OK;
}

#endif // SHANCHEN

int lb_lbfluid_set_density(double *p_dens) {
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (p_dens[ii] <= 0)
      return -1;
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      lbpar_gpu.rho[ii] = (float)p_dens[ii];
      on_lb_params_change_gpu(LBPAR_DENSITY);
#endif // LB_GPU
    } else {
#ifdef LB
      lbpar.rho = p_dens[ii];
      mpi_bcast_lb_params(LBPAR_DENSITY);
#endif // LB
    }
  }
  return 0;
}

int lb_lbfluid_set_visc(double *p_visc) {
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (p_visc[ii] <= 0)
      return -1;
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      lbpar_gpu.viscosity[ii] = (float)p_visc[ii];
      on_lb_params_change_gpu(LBPAR_VISCOSITY);
#endif // LB_GPU
    } else {
#ifdef LB
      lbpar.viscosity = p_visc[ii];
      mpi_bcast_lb_params(LBPAR_VISCOSITY);
#endif // LB
    }
  }
  return 0;
}

int lb_lbfluid_set_bulk_visc(double *p_bulk_visc) {
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (p_bulk_visc[ii] <= 0)
      return -1;
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      lbpar_gpu.bulk_viscosity[ii] = (float)p_bulk_visc[ii];
      lbpar_gpu.is_TRT = false;
      on_lb_params_change_gpu(LBPAR_BULKVISC);
#endif // LB_GPU
    } else {
#ifdef LB
      lbpar.bulk_viscosity = p_bulk_visc[ii];
      lbpar.is_TRT = false;
      mpi_bcast_lb_params(LBPAR_BULKVISC);
#endif // LB
    }
  }
  return 0;
}

int lb_lbfluid_set_gamma_odd(double *p_gamma_odd) {
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (fabs(p_gamma_odd[ii]) > 1)
      return -1;
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      lbpar_gpu.gamma_odd[ii] = (float)p_gamma_odd[ii];
      lbpar_gpu.is_TRT = false;
      on_lb_params_change_gpu(0);
#endif // LB_GPU
    } else {
#ifdef LB
      lbpar.gamma_odd = *p_gamma_odd;
      lbpar.is_TRT = false;
      mpi_bcast_lb_params(0);
#endif // LB
    }
  }
  return 0;
}

int lb_lbfluid_set_gamma_even(double *p_gamma_even) {
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (fabs(p_gamma_even[ii]) > 1)
      return -1;
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      lbpar_gpu.gamma_even[ii] = (float)p_gamma_even[ii];
      lbpar_gpu.is_TRT = false;
      on_lb_params_change_gpu(0);
#endif // LB_GPU
    } else {
#ifdef LB
      lbpar.gamma_even = *p_gamma_even;
      lbpar.is_TRT = false;
      mpi_bcast_lb_params(0);
#endif // LB
    }
  }
  return 0;
}

int lb_lbfluid_set_friction(double *p_friction) {
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (p_friction[ii] <= 0)
      return -1;
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      lbpar_gpu.friction[ii] = (float)p_friction[ii];
      on_lb_params_change_gpu(LBPAR_FRICTION);
#endif // LB_GPU
    } else {
#ifdef LB
      lbpar.friction = p_friction[ii];
      mpi_bcast_lb_params(LBPAR_FRICTION);
#endif // LB
    }
  }
  return 0;
}

int lb_lbfluid_get_friction(double *p_friction) {
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      p_friction[ii] = (double)lbpar_gpu.friction[ii];
#endif // LB_GPU
    } else {
#ifdef LB
      p_friction[ii] = lbpar.friction;
#endif // LB
    }
  }
  return 0;
}

int lb_lbfluid_set_couple_flag(int couple_flag) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    if (couple_flag != LB_COUPLE_TWO_POINT &&
        couple_flag != LB_COUPLE_THREE_POINT)
      return -1;
    lbpar_gpu.lb_couple_switch = couple_flag;
#endif // LB_GPU
  } else {
#ifdef LB
    /* Only the two point nearest neighbor coupling is present in the case of
       the cpu, so just throw an error if something else is tried */
    if (couple_flag != LB_COUPLE_TWO_POINT)
      return -1;
#endif // LB
  }
  return 0;
}

int lb_lbfluid_get_couple_flag(int *couple_flag) {
  *couple_flag = LB_COUPLE_NULL;
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    *couple_flag = lbpar_gpu.lb_couple_switch;
#endif
  } else {
#ifdef LB
    *couple_flag = LB_COUPLE_TWO_POINT;
#endif
  }
  return 0;
}

int lb_lbfluid_set_agrid(double p_agrid) {
  if (p_agrid <= 0)
    return -1;
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    lbpar_gpu.agrid = (float)p_agrid;

    lbpar_gpu.dim_x = (unsigned int)rint(box_l[0] / p_agrid);
    lbpar_gpu.dim_y = (unsigned int)rint(box_l[1] / p_agrid);
    lbpar_gpu.dim_z = (unsigned int)rint(box_l[2] / p_agrid);
    unsigned int tmp[3];
    tmp[0] = lbpar_gpu.dim_x;
    tmp[1] = lbpar_gpu.dim_y;
    tmp[2] = lbpar_gpu.dim_z;
    /* sanity checks */
    for (int dir = 0; dir < 3; dir++) {
      /* check if box_l is compatible with lattice spacing */
      if (fabs(box_l[dir] - tmp[dir] * p_agrid) > ROUND_ERROR_PREC) {
        runtimeErrorMsg() << "Lattice spacing p_agrid= " << p_agrid
                          << " is incompatible with box_l[" << dir
                          << "]=" << box_l[dir] << ", factor=" << tmp[dir]
                          << " err= " << fabs(box_l[dir] - tmp[dir] * p_agrid);
      }
    }
    lbpar_gpu.number_of_nodes =
        lbpar_gpu.dim_x * lbpar_gpu.dim_y * lbpar_gpu.dim_z;
    on_lb_params_change_gpu(LBPAR_AGRID);
#endif // LB_GPU
  } else {
#ifdef LB
    lbpar.agrid = p_agrid;
    mpi_bcast_lb_params(LBPAR_AGRID);
#endif // LB
  }
  return 0;
}

int lb_lbfluid_set_tau(double p_tau) {
  if (p_tau <= 0)
    return -1;
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    lbpar_gpu.tau = (float)p_tau;
    on_lb_params_change_gpu(0);
#endif // LB_GPU
  } else {
#ifdef LB
    lbpar.tau = p_tau;
    mpi_bcast_lb_params(0);
#endif // LB
  }
  return 0;
}

#ifdef SHANCHEN
int lb_lbfluid_set_remove_momentum(void) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    lbpar_gpu.remove_momentum = 1;
    on_lb_params_change_gpu(0);
#endif // LB_GPU
  } else {
#ifdef LB
    return -1;
#endif // LB
  }
  return 0;
}
#endif // SHANCHEN

int lb_lbfluid_set_ext_force_density(int component, double p_fx, double p_fy,
                                     double p_fz) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    if (lbpar_gpu.tau < 0.0)
      return 2;

    if (lbpar_gpu.rho[component] <= 0.0)
      return 3;

    /* external force density is stored in MD units */
    lbpar_gpu.ext_force_density[3 * component + 0] = (float)p_fx;
    lbpar_gpu.ext_force_density[3 * component + 1] = (float)p_fy;
    lbpar_gpu.ext_force_density[3 * component + 2] = (float)p_fz;
    if (p_fx != 0 || p_fy != 0 || p_fz != 0) {
      lbpar_gpu.external_force_density = 1;
    } else {
      lbpar_gpu.external_force_density = 0;
    }
    lb_reinit_extern_nodeforce_GPU(&lbpar_gpu);

#endif // LB_GPU
  } else {
#ifdef LB
    if (lbpar.tau < 0.0)
      return 2;

    if (lbpar.rho <= 0.0)
      return 3;

    lbpar.ext_force_density[0] = p_fx;
    lbpar.ext_force_density[1] = p_fy;
    lbpar.ext_force_density[2] = p_fz;
    mpi_bcast_lb_params(LBPAR_EXTFORCE);
#endif // LB
  }
  return 0;
}

int lb_lbfluid_get_density(double *p_dens) {
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      p_dens[ii] = (double)lbpar_gpu.rho[ii];
#endif // LB_GPU
    } else {
#ifdef LB
      p_dens[ii] = lbpar.rho;
#endif // LB
    }
  }
  return 0;
}

int lb_lbfluid_get_visc(double *p_visc) {
  for (int ii = 0; ii < LB_COMPONENTS; ii++) {
    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
      p_visc[ii] = (double)lbpar_gpu.viscosity[ii];
#endif // LB_GPU
    } else {
#ifdef LB
      p_visc[ii] = lbpar.viscosity;
#endif // LB
    }
  }
  return 0;
}

int lb_lbfluid_get_bulk_visc(double *p_bulk_visc) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    *p_bulk_visc = lbpar_gpu.bulk_viscosity[0];
#endif // LB_GPU
  } else {
#ifdef LB
    *p_bulk_visc = lbpar.bulk_viscosity;
#endif // LB
  }
  return 0;
}

int lb_lbfluid_get_gamma_odd(double *p_gamma_odd) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    *p_gamma_odd = lbpar_gpu.gamma_odd[0];
#endif // LB_GPU
  } else {
#ifdef LB
    *p_gamma_odd = lbpar.gamma_odd;
#endif // LB
  }
  return 0;
}

int lb_lbfluid_get_gamma_even(double *p_gamma_even) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    *p_gamma_even = lbpar_gpu.gamma_even[0];
#endif // LB_GPU
  } else {
#ifdef LB
    *p_gamma_even = lbpar.gamma_even;
#endif // LB
  }
  return 0;
}

int lb_lbfluid_get_agrid(double *p_agrid) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    *p_agrid = lbpar_gpu.agrid;
#endif // LB_GPU
  } else {
#ifdef LB
    *p_agrid = lbpar.agrid;
#endif // LB
  }
  return 0;
}

int lb_lbfluid_get_tau(double *p_tau) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    *p_tau = lbpar_gpu.tau;
#endif // LB_GPU
  } else {
#ifdef LB
    *p_tau = lbpar.tau;
#endif // LB
  }
  return 0;
}

int lb_lbfluid_get_ext_force_density(double *p_f) {
#ifdef SHANCHEN
  fprintf(stderr, "Not implemented yet (%s:%d) ", __FILE__, __LINE__);
  errexit();
#endif // SHANCHEN
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    p_f[0] = lbpar_gpu.ext_force_density[0];
    p_f[1] = lbpar_gpu.ext_force_density[1];
    p_f[2] = lbpar_gpu.ext_force_density[2];
#endif // LB_GPU
  } else {
#ifdef LB
    p_f[0] = lbpar.ext_force_density[0];
    p_f[1] = lbpar.ext_force_density[1];
    p_f[2] = lbpar.ext_force_density[2];
#endif // LB
  }
  return 0;
}

int lb_lbfluid_print_vtk_boundary(char *filename) {
#ifdef LB_ADAPTIVE
  /* strip file ending from filename (if given) */
  char *pos_file_ending;
  pos_file_ending = strrchr(filename, '.');
  if (pos_file_ending != nullptr) {
    *pos_file_ending = '\0';
  }
  int len = static_cast<int>(strlen(filename));
  ++len;

  /* call mpi printing routine on all slaves and communicate the filename */
  mpi_call(mpi_lbadapt_vtk_print_boundary, -1, len);
  MPI_Bcast(filename, len, MPI_CHAR, 0, comm_cart);

  /* perform master IO routine here. */
#ifndef LB_ADAPTIVE_GPU
  p4est_locidx_t num_cells = adapt_p4est->local_num_quadrants;
#else  // LB_ADAPTIVE_GPU
  p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  p4est_locidx_t num_cells = cells_per_patch * adapt_p4est->local_num_quadrants;
#endif // LB_ADAPTIVE_GPU
  castable_unique_ptr<sc_array_t> boundary =
      sc_array_new_size(sizeof(double), num_cells);

  lbadapt_get_boundary_values(boundary);

#ifndef LB_ADAPTIVE_GPU
  /* create VTK output context and set its parameters */
  p8est_vtk_context_t *context = p8est_vtk_context_new(adapt_p4est, filename);
  p8est_vtk_context_set_scale(context, 1); /* quadrant at almost full scale */

  /* begin writing the output files */
  context = p8est_vtk_write_header(context);
  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing vtk header");
  // clang-format off
  context = p8est_vtk_write_cell_dataf(context, 1, /* write tree indices */
                                       1, /* write the refinement level */
                                       1, /* write the mpi process id */
                                       0, /* do not wrap the mpi rank */
                                       1, /* write boundary as scalar cell
                                             data */
                                       0, /* no custom cell vector data */
                                       "boundary", boundary.get(), context);
  // clang-format on

  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing cell data");

  const int retval = p8est_vtk_write_footer(context);
  SC_CHECK_ABORT(!retval, P8EST_STRING "_vtk: Error writing footer");
#else  // LB_ADAPTIVE_GPU
  /* create VTK output context and set its parameters */
  lbadapt_vtk_context_t *context = lbadapt_vtk_context_new(filename);

  /* begin writing the output files */
  context = lbadapt_vtk_write_header(context);
  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing vtk header");
  context = lbadapt_vtk_write_cell_dataf(context, 1,
                                         /* write tree indices */
                                         1, /* write the refinement level */
                                         1, /* write the mpi process id */
                                         0, /* do not wrap the mpi rank */
                                         1, /* write qid */
                                         1, /* write boundary index as scalar
                                               cell data */
                                         0, /* no custom cell vector data */
                                         "boundary", boundary.get(), context);

  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing cell data");

  const int retval = lbadapt_vtk_write_footer(context);
  SC_CHECK_ABORT(!retval, P8EST_STRING "_vtk: Error writing footer");
#endif // LB_ADAPTIVE_GPU
#else  // LB_ADAPTIVE
  FILE *fp = fopen(filename, "w");

  if (fp == nullptr) {
    return 1;
  }

  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    unsigned int *bound_array;
    bound_array = (unsigned int *)Utils::malloc(lbpar_gpu.number_of_nodes *
                                                sizeof(unsigned int));
    lb_get_boundary_flags_GPU(bound_array);

    int j;
    /** print of the calculated phys values */
    fprintf(fp,
            "# vtk DataFile Version 2.0\nlbboundaries\n"
            "ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\n"
            "ORIGIN %f %f %f\nSPACING %f %f %f\nPOINT_DATA %u\n"
            "SCALARS boundary float 1\nLOOKUP_TABLE default\n",
            lbpar_gpu.dim_x, lbpar_gpu.dim_y, lbpar_gpu.dim_z,
            lbpar_gpu.agrid * 0.5, lbpar_gpu.agrid * 0.5, lbpar_gpu.agrid * 0.5,
            lbpar_gpu.agrid, lbpar_gpu.agrid, lbpar_gpu.agrid,
            lbpar_gpu.number_of_nodes);
    for (j = 0; j < int(lbpar_gpu.number_of_nodes); ++j) {
      /** print of the calculated phys values */
      fprintf(fp, "%d \n", bound_array[j]);
    }
    free(bound_array);
#endif // LB_GPU
  } else {
#ifdef LB
    Vector3i pos;
    int boundary;
    int gridsize[3];

    gridsize[0] = box_l[0] / lbpar.agrid;
    gridsize[1] = box_l[1] / lbpar.agrid;
    gridsize[2] = box_l[2] / lbpar.agrid;

    fprintf(fp,
            "# vtk DataFile Version 2.0\nlbboundaries\n"
            "ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\n"
            "ORIGIN %f %f %f\nSPACING %f %f %f\nPOINT_DATA %d\n"
            "SCALARS boundary float 1\nLOOKUP_TABLE default\n",
            gridsize[0], gridsize[1], gridsize[2], lblattice.agrid[0] * 0.5,
            lblattice.agrid[1] * 0.5, lblattice.agrid[2] * 0.5,
            lblattice.agrid[0], lblattice.agrid[1], lblattice.agrid[2],
            gridsize[0] * gridsize[1] * gridsize[2]);

    for (pos[2] = 0; pos[2] < gridsize[2]; pos[2]++) {
      for (pos[1] = 0; pos[1] < gridsize[1]; pos[1]++) {
        for (pos[0] = 0; pos[0] < gridsize[0]; pos[0]++) {
          lb_lbnode_get_boundary(pos, &boundary);
          fprintf(fp, "%d \n", boundary);
        }
      }
    }
#endif // LB
  }
  fclose(fp);
#endif // LB_ADAPTIVE
  return 0;
}

int lb_lbfluid_print_vtk_density(char **filename) {
#ifdef LB_ADAPTIVE
  /* strip file ending from filename (if given) */
  char *pos_file_ending;
  pos_file_ending = strrchr(*filename, '.');
  if (pos_file_ending != nullptr) {
    *pos_file_ending = '\0';
  }
  int len = static_cast<int>(strlen(*filename));
  ++len;

  /* call mpi printing routine on all slaves and communicate the filename */
  mpi_call(mpi_lbadapt_vtk_print_density, -1, len);
  MPI_Bcast(*filename, len, MPI_CHAR, 0, comm_cart);

  /* perform master IO routine here. */
  /* TODO: move this to communication? */

#ifndef LB_ADAPTIVE_GPU
  p4est_locidx_t num_cells = adapt_p4est->local_num_quadrants;
#else  // LB_ADAPTIVE_GPU
  p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  p4est_locidx_t num_cells = cells_per_patch * adapt_p4est->local_num_quadrants;
#endif // LB_ADAPTIVE_GPU
  castable_unique_ptr<sc_array_t> density =
      sc_array_new_size(sizeof(double), num_cells);

  lbadapt_get_density_values(density);

#ifndef LB_ADAPTIVE_GPU
  /* create VTK output context and set its parameters */
  p8est_vtk_context_t *context = p8est_vtk_context_new(adapt_p4est, *filename);
  p8est_vtk_context_set_scale(context, 1); /* quadrant at full scale */

  /* begin writing the output files */
  context = p8est_vtk_write_header(context);
  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing vtk header");
  // clang-format off
  context = p8est_vtk_write_cell_dataf(context,
                                       1, /* write tree indices */
                                       1, /* write the refinement level */
                                       1, /* write the mpi process id */
                                       0, /* do not wrap the mpi rank */
                                       1, /* write density as scalar cell
                                             data */
                                       0, /* no custom cell vector data */
                                       "density", density.get(), context);
  // clang-format on

  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing cell data");

  const int retval = p8est_vtk_write_footer(context);
  SC_CHECK_ABORT(!retval, P8EST_STRING "_vtk: Error writing footer");
#else  // LB_ADAPTIVE_GPU
  /* create VTK output context and set its parameters */
  lbadapt_vtk_context_t *context = lbadapt_vtk_context_new(filename);

  /* begin writing the output files */
  context = lbadapt_vtk_write_header(context);
  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing vtk header");
  context = lbadapt_vtk_write_cell_dataf(context, 1, /* write tree indices */
                                         1, /* write the refinement level */
                                         1, /* write the mpi process id */
                                         0, /* do not wrap the mpi rank */
                                         1, /* write qid */
                                         1, /* write density as scalar cell
                                               data */
                                         0, /* no custom cell vector data */
                                         "density", density.get(), context);

  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing cell data");

  const int retval = lbadapt_vtk_write_footer(context);
  SC_CHECK_ABORT(!retval, P8EST_STRING "_vtk: Error writing footer");
#endif // LB_ADAPTIVE_GPU
#else  // LB_ADAPTIVE
  int ii;

  for (ii = 0; ii < LB_COMPONENTS; ++ii) {
    FILE *fp = fopen(filename[ii], "w");

    if (fp == nullptr) {
      perror("lb_lbfluid_print_vtk_density");
      return 1;
    }

    if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU

      int j;
      size_t size_of_values =
          lbpar_gpu.number_of_nodes * sizeof(LB_rho_v_pi_gpu);
      host_values = (LB_rho_v_pi_gpu *)Utils::malloc(size_of_values);
      lb_get_values_GPU(host_values);

      fprintf(fp,
              "# vtk DataFile Version 2.0\nlbfluid_gpu\nASCII\nDATASET "
              "STRUCTURED_POINTS\nDIMENSIONS %u %u %u\nORIGIN %f %f "
              "%f\nSPACING %f %f %f\nPOINT_DATA %u\nSCALARS density float "
              "1\nLOOKUP_TABLE default\n",
              lbpar_gpu.dim_x, lbpar_gpu.dim_y, lbpar_gpu.dim_z,
              lbpar_gpu.agrid * 0.5, lbpar_gpu.agrid * 0.5,
              lbpar_gpu.agrid * 0.5, lbpar_gpu.agrid, lbpar_gpu.agrid,
              lbpar_gpu.agrid, lbpar_gpu.number_of_nodes);

      for (j = 0; j < int(lbpar_gpu.number_of_nodes); ++j) {
        /** print the calculated phys values */
        fprintf(fp, "%f\n", host_values[j].rho[ii]);
      }
      free(host_values);

#endif // LB_GPU
    } else {
#ifdef LB
      fprintf(stderr, "Not implemented yet (%s:%d) ", __FILE__, __LINE__);
      errexit();
#endif // LB
    }
    fclose(fp);
  }
#endif // LB_ADAPTIVE
  return 0;
}

int lb_lbfluid_print_vtk_velocity(char *filename, std::vector<int> bb1,
                                  std::vector<int> bb2) {
#ifdef LB_ADAPTIVE
  /* strip file ending from filename (if given) */
  char *pos_file_ending;
  pos_file_ending = strrchr(filename, '.');
  if (pos_file_ending != nullptr) {
    *pos_file_ending = '\0';
  }
  int len = static_cast<int>(strlen(filename));
  ++len;

  /* call mpi printing routine on all slaves and communicate the filename */
  mpi_call(mpi_lbadapt_vtk_print_velocity, -1, len);
  MPI_Bcast(filename, len, MPI_CHAR, 0, comm_cart);

  /* perform master IO routine here. */
  /* TODO: move this to communication? */

#ifndef LB_ADAPTIVE_GPU
  p4est_locidx_t num_cells = adapt_p4est->local_num_quadrants;
#else  // LB_ADAPTIVE_GPU
  p4est_locidx_t cells_per_patch =
      LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE * LBADAPT_PATCHSIZE;
  p4est_locidx_t num_cells = cells_per_patch * adapt_p4est->local_num_quadrants;
#endif // LB_ADAPTIVE_GPU
  castable_unique_ptr<sc_array_t> velocity =
      sc_array_new_size(sizeof(double), P8EST_DIM * num_cells);
  castable_unique_ptr<sc_array_t> vel_pts =
      sc_array_new_size(sizeof(double), P8EST_CHILDREN * P8EST_DIM * num_cells);
  castable_unique_ptr<sc_array_t> vorticity =
      sc_array_new_size(sizeof(double), P8EST_DIM * num_cells);

  lbadapt_get_velocity_values(velocity);
  lbadapt_get_velocity_values_nodes(vel_pts);
  lbadapt_get_vorticity_values(vorticity);

#ifndef LB_ADAPTIVE_GPU
  /* create VTK output context and set its parameters */
  p8est_vtk_context_t *context = p8est_vtk_context_new(adapt_p4est, filename);
  p8est_vtk_context_set_scale(context, 1);

  /* begin writing the output files */
  context = p8est_vtk_write_header(context);
  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing vtk header");

  // clang-format off
  context = p8est_vtk_write_cell_dataf(context,
                                       1, /* write tree indices */
                                       1, /* write the refinement level */
                                       1, /* write the mpi process id */
                                       0, /* do not wrap the mpi rank */
                                       0, /* no custom cell scalar data */
                                       2, /* write velocities as cell vector
                                             data */
                                       "vorticity", vorticity.get(),
                                       "velocity", velocity.get(), context);
  // clang-format on
  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing cell data");

  context = p8est_vtk_write_point_dataf(context, 0, 1, "velocity node",
                                        vel_pts.get(), context);
  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing cell data");

  const int retval = p8est_vtk_write_footer(context);
  SC_CHECK_ABORT(!retval, P8EST_STRING "_vtk: Error writing footer");
#else  // LB_ADAPTIVE_GPU
  /* create VTK output context and set its parameters */
  lbadapt_vtk_context_t *context = lbadapt_vtk_context_new(filename);

  /* begin writing the output files */
  context = lbadapt_vtk_write_header(context);
  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing vtk header");

  context = lbadapt_vtk_write_cell_dataf(context, 1, /* write tree indices */
                                         1, /* write the refinement level */
                                         1, /* write the mpi process id */
                                         0, /* do not wrap the mpi rank */
                                         1, /* write qid */
                                         0, /* no custom cell scalar data */
                                         1, /* write velocities as cell vector
                                               data */
                                         "velocity", velocity.get(), context);

  SC_CHECK_ABORT(context != nullptr,
                 P8EST_STRING "_vtk: Error writing cell data");

  const int retval = lbadapt_vtk_write_footer(context);
#endif // LB_ADAPTIVE_GPU
#else  // LB_ADAPTIVE
  FILE *fp = fopen(filename, "w");

  if (fp == nullptr) {
    return 1;
  }

  std::vector<int> bb_low;
  std::vector<int> bb_high;

  for (std::vector<int>::iterator val1 = bb1.begin(), val2 = bb2.begin();
       val1 != bb1.end() && val2 != bb2.end(); ++val1, ++val2) {
    if (*val1 == -1 || *val2 == -1) {
      bb_low = {0, 0, 0};
      if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
        bb_high = {static_cast<int>(lbpar_gpu.dim_x) - 1,
                   static_cast<int>(lbpar_gpu.dim_y) - 1,
                   static_cast<int>(lbpar_gpu.dim_z) - 1};
#endif // LB_GPU
      } else {
#ifdef LB
        bb_high = {lblattice.global_grid[0] - 1, lblattice.global_grid[1] - 1,
                   lblattice.global_grid[2] - 1};
#endif // LB
      }
      break;
    }

    bb_low.push_back(std::min(*val1, *val2));
    bb_high.push_back(std::max(*val1, *val2));
  }

  Vector3i pos;
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    size_t size_of_values = lbpar_gpu.number_of_nodes * sizeof(LB_rho_v_pi_gpu);
    host_values = (LB_rho_v_pi_gpu *)Utils::malloc(size_of_values);
    lb_get_values_GPU(host_values);
    fprintf(fp,
            "# vtk DataFile Version 2.0\nlbfluid_gpu\n"
            "ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\n"
            "ORIGIN %f %f %f\nSPACING %f %f %f\nPOINT_DATA %d\n"
            "SCALARS velocity float 3\nLOOKUP_TABLE default\n",
            bb_high[0] - bb_low[0] + 1, bb_high[1] - bb_low[1] + 1,
            bb_high[2] - bb_low[2] + 1, (bb_low[0] + 0.5) * lbpar_gpu.agrid,
            (bb_low[1] + 0.5) * lbpar_gpu.agrid,
            (bb_low[2] + 0.5) * lbpar_gpu.agrid, lbpar_gpu.agrid,
            lbpar_gpu.agrid, lbpar_gpu.agrid,
            (bb_high[0] - bb_low[0] + 1) * (bb_high[1] - bb_low[1] + 1) *
                (bb_high[2] - bb_low[2] + 1));
    for (pos[2] = bb_low[2]; pos[2] <= bb_high[2]; pos[2]++)
      for (pos[1] = bb_low[1]; pos[1] <= bb_high[1]; pos[1]++)
        for (pos[0] = bb_low[0]; pos[0] <= bb_high[0]; pos[0]++) {
          int j = lbpar_gpu.dim_y * lbpar_gpu.dim_x * pos[2] +
                  lbpar_gpu.dim_x * pos[1] + pos[0];
          fprintf(fp, "%f %f %f\n", host_values[j].v[0], host_values[j].v[1],
                  host_values[j].v[2]);
        }
    free(host_values);
#endif // LB_GPU
  } else {
#ifdef LB
    double u[3];

    fprintf(fp,
            "# vtk DataFile Version 2.0\nlbfluid_cpu\n"
            "ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\n"
            "ORIGIN %f %f %f\nSPACING %f %f %f\nPOINT_DATA %d\n"
            "SCALARS velocity float 3\nLOOKUP_TABLE default\n",
            bb_high[0] - bb_low[0] + 1, bb_high[1] - bb_low[1] + 1,
            bb_high[2] - bb_low[2] + 1, (bb_low[0] + 0.5) * lblattice.agrid[0],
            (bb_low[1] + 0.5) * lblattice.agrid[1],
            (bb_low[2] + 0.5) * lblattice.agrid[2], lblattice.agrid[0],
            lblattice.agrid[1], lblattice.agrid[2],
            (bb_high[0] - bb_low[0] + 1) * (bb_high[1] - bb_low[1] + 1) *
                (bb_high[2] - bb_low[2] + 1));

    for (pos[2] = bb_low[2]; pos[2] <= bb_high[2]; pos[2]++)
      for (pos[1] = bb_low[1]; pos[1] <= bb_high[1]; pos[1]++)
        for (pos[0] = bb_low[0]; pos[0] <= bb_high[0]; pos[0]++) {
          lb_lbnode_get_u(pos, u);
          fprintf(fp, "%f %f %f\n", u[0], u[1], u[2]);
        }
#endif // LB
  }
  fclose(fp);
#endif // LB_ADAPTIVE
  return 0;
}

int lb_lbfluid_print_boundary(char *filename) {
#ifndef LB_ADAPTIVE
  FILE *fp = fopen(filename, "w");

  if (fp == nullptr) {
    return 1;
  }

  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    unsigned int *bound_array;
    bound_array = (unsigned int *)Utils::malloc(lbpar_gpu.number_of_nodes *
                                                sizeof(unsigned int));
    lb_get_boundary_flags_GPU(bound_array);

    int xyz[3];
    int j;
    for (j = 0; j < int(lbpar_gpu.number_of_nodes); ++j) {
      xyz[0] = j % lbpar_gpu.dim_x;
      int k = j / lbpar_gpu.dim_x;
      xyz[1] = k % lbpar_gpu.dim_y;
      k /= lbpar_gpu.dim_y;
      xyz[2] = k;
      /** print of the calculated phys values */
      fprintf(fp, "%f %f %f %u\n", (xyz[0] + 0.5) * lbpar_gpu.agrid,
              (xyz[1] + 0.5) * lbpar_gpu.agrid,
              (xyz[2] + 0.5) * lbpar_gpu.agrid, bound_array[j]);
    }
    free(bound_array);
#endif // LB_GPU
  } else {
#ifdef LB
    Vector3i pos;
    int boundary;
    int gridsize[3];

    gridsize[0] = box_l[0] / lblattice.agrid[0];
    gridsize[1] = box_l[1] / lblattice.agrid[1];
    gridsize[2] = box_l[2] / lblattice.agrid[2];

    for (pos[2] = 0; pos[2] < gridsize[2]; pos[2]++) {
      for (pos[1] = 0; pos[1] < gridsize[1]; pos[1]++) {
        for (pos[0] = 0; pos[0] < gridsize[0]; pos[0]++) {
          lb_lbnode_get_boundary(pos, &boundary);
          boundary = (boundary != 0 ? 1 : 0);
          fprintf(fp, "%f %f %f %d\n", (pos[0] + 0.5) * lblattice.agrid[0],
                  (pos[1] + 0.5) * lblattice.agrid[1],
                  (pos[2] + 0.5) * lblattice.agrid[2], boundary);
        }
      }
    }
#endif // LB
  }

  fclose(fp);
#endif // LB_ADAPTIVE
  return 0;
}

int lb_lbfluid_print_velocity(char *filename) {
  FILE *fp = fopen(filename, "w");

  if (fp == nullptr) {
    return 1;
  }

  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
#ifdef SHANCHEN
    fprintf(stderr, "TODO:adapt for SHANCHEN (%s:%d)\n", __FILE__, __LINE__);
    errexit();
#endif // SHANCHEN
    size_t size_of_values = lbpar_gpu.number_of_nodes * sizeof(LB_rho_v_pi_gpu);
    host_values = (LB_rho_v_pi_gpu *)Utils::malloc(size_of_values);
    lb_get_values_GPU(host_values);
    int xyz[3];
    int j;
    for (j = 0; j < int(lbpar_gpu.number_of_nodes); ++j) {
      xyz[0] = j % lbpar_gpu.dim_x;
      int k = j / lbpar_gpu.dim_x;
      xyz[1] = k % lbpar_gpu.dim_y;
      k /= lbpar_gpu.dim_y;
      xyz[2] = k;
      /** print of the calculated phys values */
      fprintf(fp, "%f %f %f %f %f %f\n", (xyz[0] + 0.5) * lbpar_gpu.agrid,
              (xyz[1] + 0.5) * lbpar_gpu.agrid,
              (xyz[2] + 0.5) * lbpar_gpu.agrid, host_values[j].v[0],
              host_values[j].v[1], host_values[j].v[2]);
    }
    free(host_values);
#endif // LB_GPU
  } else {
#ifdef LB
#ifndef LB_ADAPTIVE
    Vector3i pos;
    double u[3];
    int gridsize[3];

    gridsize[0] = box_l[0] / lblattice.agrid[0];
    gridsize[1] = box_l[1] / lblattice.agrid[1];
    gridsize[2] = box_l[2] / lblattice.agrid[2];

    for (pos[2] = 0; pos[2] < gridsize[2]; pos[2]++) {
      for (pos[1] = 0; pos[1] < gridsize[1]; pos[1]++) {
        for (pos[0] = 0; pos[0] < gridsize[0]; pos[0]++) {
#ifdef SHANCHEN
          fprintf(stderr, "SHANCHEN not implemented for the CPU LB\n", __FILE__,
                  __LINE__);
          errexit();
#endif // SHANCHEN
          lb_lbnode_get_u(pos, u);
          fprintf(fp, "%f %f %f %f %f %f\n",
                  (pos[0] + 0.5) * lblattice.agrid[0],
                  (pos[1] + 0.5) * lblattice.agrid[1],
                  (pos[2] + 0.5) * lblattice.agrid[2], u[0], u[1], u[2]);
        }
      }
    }
#endif // LB_ADAPTIVE
#endif // LB
  }

  fclose(fp);
  return 0;
}

int lb_lbfluid_save_checkpoint(char *filename, int binary) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    FILE *cpfile;
    cpfile = fopen(filename, "w");
    if (!cpfile) {
      return ES_ERROR;
    }
    float *host_checkpoint_vd =
        (float *)Utils::malloc(lbpar_gpu.number_of_nodes * 19 * sizeof(float));
    unsigned int *host_checkpoint_boundary = (unsigned int *)Utils::malloc(
        lbpar_gpu.number_of_nodes * sizeof(unsigned int));
    lbForceFloat *host_checkpoint_force = (lbForceFloat *)Utils::malloc(
        lbpar_gpu.number_of_nodes * 3 * sizeof(lbForceFloat));
    uint64_t host_checkpoint_philox_counter;
    lb_save_checkpoint_GPU(host_checkpoint_vd, host_checkpoint_boundary,
                           host_checkpoint_force,
                           &host_checkpoint_philox_counter);
    if (!binary) {
      for (int n = 0; n < (19 * int(lbpar_gpu.number_of_nodes)); n++) {
        fprintf(cpfile, "%.8E \n", host_checkpoint_vd[n]);
      }
      for (int n = 0; n < int(lbpar_gpu.number_of_nodes); n++) {
        fprintf(cpfile, "%u \n", host_checkpoint_boundary[n]);
      }
      for (int n = 0; n < (3 * int(lbpar_gpu.number_of_nodes)); n++) {
        fprintf(cpfile, "%.8E \n", host_checkpoint_force[n]);
      }
      fprintf(cpfile, "%" PRIu64 "\n", host_checkpoint_philox_counter);
    } else {
      fwrite(host_checkpoint_vd, sizeof(float),
             19 * int(lbpar_gpu.number_of_nodes), cpfile);
      fwrite(host_checkpoint_boundary, sizeof(int),
             int(lbpar_gpu.number_of_nodes), cpfile);
      fwrite(host_checkpoint_force, sizeof(lbForceFloat),
             3 * int(lbpar_gpu.number_of_nodes), cpfile);
      fwrite(&host_checkpoint_philox_counter, sizeof(uint64_t), 1, cpfile);
    }
    fclose(cpfile);
    free(host_checkpoint_vd);
    free(host_checkpoint_boundary);
    free(host_checkpoint_force);
#endif // LB_GPU
  } else if (lattice_switch & LATTICE_LB) {
#ifdef LB
    FILE *cpfile;
    cpfile = fopen(filename, "w");
    if (!cpfile) {
      return ES_ERROR;
    }
    std::array<double, 19> pop;
    Vector3i ind;

    int gridsize[3];

    gridsize[0] = box_l[0] / lbpar.agrid;
    gridsize[1] = box_l[1] / lbpar.agrid;
    gridsize[2] = box_l[2] / lbpar.agrid;

    for (int i = 0; i < gridsize[0]; i++) {
      for (int j = 0; j < gridsize[1]; j++) {
        for (int k = 0; k < gridsize[2]; k++) {
          ind[0] = i;
          ind[1] = j;
          ind[2] = k;
          lb_lbnode_get_pop(ind, pop.data());
          if (!binary) {
            for (int n = 0; n < 19; n++) {
              fprintf(cpfile, "%.16e ", pop[n]);
            }
            fprintf(cpfile, "\n");
          } else {
            fwrite(pop.data(), sizeof(double), 19, cpfile);
          }
        }
      }
    }
    fclose(cpfile);
#endif // LB
  }
  return ES_OK;
}

int lb_lbfluid_load_checkpoint(char *filename, int binary) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    FILE *cpfile;
    cpfile = fopen(filename, "r");
    if (!cpfile) {
      return ES_ERROR;
    }
    std::vector<float> host_checkpoint_vd(lbpar_gpu.number_of_nodes * 19);
    std::vector<unsigned int> host_checkpoint_boundary(
        lbpar_gpu.number_of_nodes);
    std::vector<lbForceFloat> host_checkpoint_force(lbpar_gpu.number_of_nodes *
                                                    3);
    uint64_t host_checkpoint_philox_counter;
    int res;
    if (!binary) {
      for (int n = 0; n < (19 * int(lbpar_gpu.number_of_nodes)); n++) {
        res = fscanf(cpfile, "%f", &host_checkpoint_vd[n]);
      }
      for (int n = 0; n < int(lbpar_gpu.number_of_nodes); n++) {
        res = fscanf(cpfile, "%u", &host_checkpoint_boundary[n]);
      }
      for (int n = 0; n < (3 * int(lbpar_gpu.number_of_nodes)); n++) {
        res = fscanf(cpfile, "%f", &host_checkpoint_force[n]);
      }
      res = fscanf(cpfile, "%" SCNu64, &host_checkpoint_philox_counter);
      if (res == EOF)
        throw std::runtime_error("Error while reading LB checkpoint.");
    } else {
      if (fread(host_checkpoint_vd.data(), sizeof(float),
                19 * int(lbpar_gpu.number_of_nodes),
                cpfile) != (unsigned int)(19 * lbpar_gpu.number_of_nodes))
        return ES_ERROR;
      if (fread(host_checkpoint_boundary.data(), sizeof(int),
                int(lbpar_gpu.number_of_nodes),
                cpfile) != (unsigned int)lbpar_gpu.number_of_nodes) {
        fclose(cpfile);
        return ES_ERROR;
      }
      if (fread(host_checkpoint_force.data(), sizeof(lbForceFloat),
                3 * int(lbpar_gpu.number_of_nodes),
                cpfile) != (unsigned int)(3 * lbpar_gpu.number_of_nodes)) {
        fclose(cpfile);
        return ES_ERROR;
      }
      if (fread(&host_checkpoint_philox_counter, sizeof(uint64_t), 1, cpfile) !=
          1) {
        fclose(cpfile);
        return ES_ERROR;
      }
    }
    lb_load_checkpoint_GPU(
        host_checkpoint_vd.data(), host_checkpoint_boundary.data(),
        host_checkpoint_force.data(), &host_checkpoint_philox_counter);
    fclose(cpfile);
#endif // LB_GPU
  } else if (lattice_switch & LATTICE_LB) {
#ifdef LB
    FILE *cpfile;
    cpfile = fopen(filename, "r");
    if (!cpfile) {
      return ES_ERROR;
    }
    double pop[19];
    Vector3i ind;

    int gridsize[3];
    lbpar.resend_halo = 1;
    mpi_bcast_lb_params(0);
    gridsize[0] = box_l[0] / lbpar.agrid;
    gridsize[1] = box_l[1] / lbpar.agrid;
    gridsize[2] = box_l[2] / lbpar.agrid;

    for (int i = 0; i < gridsize[0]; i++) {
      for (int j = 0; j < gridsize[1]; j++) {
        for (int k = 0; k < gridsize[2]; k++) {
          ind[0] = i;
          ind[1] = j;
          ind[2] = k;
          if (!binary) {
            if (fscanf(cpfile,
                       "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
                       "%lf %lf %lf %lf %lf %lf \n",
                       &pop[0], &pop[1], &pop[2], &pop[3], &pop[4], &pop[5],
                       &pop[6], &pop[7], &pop[8], &pop[9], &pop[10], &pop[11],
                       &pop[12], &pop[13], &pop[14], &pop[15], &pop[16],
                       &pop[17], &pop[18]) != 19) {
              return ES_ERROR;
            }
          } else {
            if (fread(pop, sizeof(double), 19, cpfile) != 19)
              return ES_ERROR;
          }
          lb_lbnode_set_pop(ind, pop);
        }
      }
    }
    fclose(cpfile);
//  lbpar.resend_halo=1;
//  mpi_bcast_lb_params(0);
#endif // LB
  } else {
    runtimeErrorMsg() << "To load an LB checkpoint one needs to have already "
                         "initialized the LB fluid with the same grid size.";
    return ES_ERROR;
  }
  return ES_OK;
}

Vector3i lb_lbfluid_get_node_state() {
  mpi_call(mpi_get_node_state_slave, -1, 0);
  return mpi_get_node_state();
}

bool lb_lbnode_is_index_valid(const Vector3i &ind) {
  auto within_bounds = [](const Vector3i &ind, const Vector3i &limits) {
    return ind < limits && ind >= Vector3i{};
  };
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    return within_bounds(ind, {static_cast<int>(lbpar_gpu.dim_x),
                               static_cast<int>(lbpar_gpu.dim_y),
                               static_cast<int>(lbpar_gpu.dim_z)});
#endif
  } else if (lattice_switch & LATTICE_LB) {
#ifdef LB
#ifndef LB_ADAPTIVE
    return within_bounds(ind, lblattice.global_grid);
#endif
#endif
  }
  return false;
}

int lb_lbnode_get_rho(const Vector3i &ind, double *p_rho) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    int single_nodeindex = ind[0] + ind[1] * lbpar_gpu.dim_x +
                           ind[2] * lbpar_gpu.dim_x * lbpar_gpu.dim_y;
    static LB_rho_v_pi_gpu *host_print_values = nullptr;

    if (host_print_values == nullptr)
      host_print_values =
          (LB_rho_v_pi_gpu *)Utils::malloc(sizeof(LB_rho_v_pi_gpu));
    lb_print_node_GPU(single_nodeindex, host_print_values);
    for (int ii = 0; ii < LB_COMPONENTS; ii++) {
      p_rho[ii] = (double)(host_print_values->rho[ii]);
    }
#endif // LB_GPU
  } else {
#ifdef LB
#ifndef LB_ADAPTIVE
    Lattice::index_t index;
    int node, grid[3], ind_shifted[3];
    double rho;
    double j[3];
    double pi[6];

    ind_shifted[0] = ind[0];
    ind_shifted[1] = ind[1];
    ind_shifted[2] = ind[2];
    node = lblattice.map_lattice_to_node(ind_shifted, grid);
    index = get_linear_index(ind_shifted[0], ind_shifted[1], ind_shifted[2],
                             lblattice.halo_grid);

    mpi_recv_fluid(node, index, &rho, j, pi);
    // unit conversion
    rho *= 1 / lbpar.agrid / lbpar.agrid / lbpar.agrid;
    *p_rho = rho;
#else  // !LB_ADAPTIVE
    double rho;
    double j[3];
    double pi[6];
    int64_t index = p4est_utils_cell_morton_idx(ind[0], ind[1], ind[2]);
    int proc = p4est_utils_idx_to_proc(forest_order::adaptive_LB, index);
    lb_float h_max = p4est_params.h[p4est_params.max_ref_level];
    mpi_recv_fluid(proc, index, &rho, j, pi);
    // unit conversion
    rho *= 1 / h_max / h_max / h_max;
    *p_rho = rho;
#endif // !LB_ADAPTIVE
#endif // LB
  }
  return 0;
}

int lb_lbnode_get_u(const Vector3i &ind, double *p_u) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    static LB_rho_v_pi_gpu *host_print_values = nullptr;
    if (host_print_values == nullptr)
      host_print_values =
          (LB_rho_v_pi_gpu *)Utils::malloc(sizeof(LB_rho_v_pi_gpu));

    int single_nodeindex = ind[0] + ind[1] * lbpar_gpu.dim_x +
                           ind[2] * lbpar_gpu.dim_x * lbpar_gpu.dim_y;
    lb_print_node_GPU(single_nodeindex, host_print_values);

    p_u[0] = (double)(host_print_values[0].v[0]);
    p_u[1] = (double)(host_print_values[0].v[1]);
    p_u[2] = (double)(host_print_values[0].v[2]);
#endif // LB_GPU
  } else {
#ifdef LB
#ifndef LB_ADAPTIVE
    Lattice::index_t index;
    int node, grid[3], ind_shifted[3];
    double rho;
    double j[3];
    double pi[6];

    ind_shifted[0] = ind[0];
    ind_shifted[1] = ind[1];
    ind_shifted[2] = ind[2];
    node = lblattice.map_lattice_to_node(ind_shifted, grid);
    index = get_linear_index(ind_shifted[0], ind_shifted[1], ind_shifted[2],
                             lblattice.halo_grid);

    mpi_recv_fluid(node, index, &rho, j, pi);
    // unit conversion
    p_u[0] = j[0] / rho * lbpar.agrid / lbpar.tau;
    p_u[1] = j[1] / rho * lbpar.agrid / lbpar.tau;
    p_u[2] = j[2] / rho * lbpar.agrid / lbpar.tau;
#else  // !LB_ADAPTIVE
    double rho;
    double j[3];
    double pi[6];
    int64_t index = p4est_utils_cell_morton_idx(ind[0], ind[1], ind[2]);
    int proc = p4est_utils_idx_to_proc(forest_order::adaptive_LB, index);
    lb_float h_max = p4est_params.h[p4est_params.max_ref_level];
    mpi_recv_fluid(proc, index, &rho, j, pi);
    // unit conversion
    p_u[0] = j[0] / rho * h_max / lbpar.tau;
    p_u[1] = j[1] / rho * h_max / lbpar.tau;
    p_u[2] = j[2] / rho * h_max / lbpar.tau;
#endif // !LB_ADAPTIVE
#endif // LB
  }
  return 0;
}

/** calculates the fluid velocity at a given position of the
 * lattice. Note that it can lead to undefined behavior if the
 * position is not within the local lattice. This version of the function
 * can be called without the position needing to be on the local processor.
 * Note that this gives a slightly different version than the values used to
 * couple to MD beads when near a wall, see
 * lb_lbfluid_get_interpolated_velocity.
 */
int lb_lbfluid_get_interpolated_velocity_global(Vector3d &p, double *v) {
#ifdef LB_ADAPTIVE
  int im[3] = {0, 0, 0}; /* dummy */
  fold_position(p, im);
  int node = p4est_utils_pos_to_proc(forest_order::adaptive_LB, p.data());
  mpi_recv_interpolated_velocity(node, p.data(), v);
#else // LB_ADAPTIVE
  double local_v[3] = {0, 0, 0},
         delta[6]{}; // velocity field, relative positions to surrounding nodes
  Vector3i ind = {0, 0, 0}, tmpind; // node indices
  int x, y, z;                      // counters

  // convert the position into lower left grid point
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    Lattice::map_position_to_lattice_global(p, ind, delta, lbpar_gpu.agrid);
#endif // LB_GPU
  } else {
#ifdef LB
    Lattice::map_position_to_lattice_global(p, ind, delta, lbpar.agrid);
#endif // LB
  }

  // set the initial velocity to zero in all directions
  v[0] = 0;
  v[1] = 0;
  v[2] = 0;

  for (z = 0; z < 2; z++) {
    for (y = 0; y < 2; y++) {
      for (x = 0; x < 2; x++) {
        // give the index of the neighbouring nodes
        tmpind[0] = ind[0] + x;
        tmpind[1] = ind[1] + y;
        tmpind[2] = ind[2] + z;

        if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
          if (tmpind[0] == int(lbpar_gpu.dim_x))
            tmpind[0] = 0;
          if (tmpind[1] == int(lbpar_gpu.dim_y))
            tmpind[1] = 0;
          if (tmpind[2] == int(lbpar_gpu.dim_z))
            tmpind[2] = 0;
#endif // LB_GPU
        } else {
#ifdef LB
          if (tmpind[0] == box_l[0] / lbpar.agrid)
            tmpind[0] = 0;
          if (tmpind[1] == box_l[1] / lbpar.agrid)
            tmpind[1] = 0;
          if (tmpind[2] == box_l[2] / lbpar.agrid)
            tmpind[2] = 0;
#endif // LB
        }

        lb_lbnode_get_u(tmpind, local_v);

        v[0] +=
            delta[3 * x + 0] * delta[3 * y + 1] * delta[3 * z + 2] * local_v[0];
        v[1] +=
            delta[3 * x + 0] * delta[3 * y + 1] * delta[3 * z + 2] * local_v[1];
        v[2] +=
            delta[3 * x + 0] * delta[3 * y + 1] * delta[3 * z + 2] * local_v[2];
      }
    }
  }
#endif // LB_ADAPTIVE

  return 0;
}

int lb_lbnode_get_pi(const Vector3i &ind, double *p_pi) {
  double p0 = 0;

  lb_lbnode_get_pi_neq(ind, p_pi);

  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    for (int ii = 0; ii < LB_COMPONENTS; ii++) {
      p0 += lbpar_gpu.rho[ii] * lbpar_gpu.agrid * lbpar_gpu.agrid /
            lbpar_gpu.tau / lbpar_gpu.tau / 3.;
    }
#endif // LB_GPU
  } else {
#ifdef LB
    p0 = lbpar.rho * lbpar.agrid * lbpar.agrid / lbpar.tau / lbpar.tau / 3.;
#endif // LB
  }

  p_pi[0] += p0;
  p_pi[2] += p0;
  p_pi[5] += p0;

  return 0;
}

int lb_lbnode_get_pi_neq(const Vector3i &ind, double *p_pi) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    static LB_rho_v_pi_gpu *host_print_values = nullptr;
    if (host_print_values == nullptr)
      host_print_values =
          (LB_rho_v_pi_gpu *)Utils::malloc(sizeof(LB_rho_v_pi_gpu));

    int single_nodeindex = ind[0] + ind[1] * lbpar_gpu.dim_x +
                           ind[2] * lbpar_gpu.dim_x * lbpar_gpu.dim_y;
    lb_print_node_GPU(single_nodeindex, host_print_values);
    for (int i = 0; i < 6; i++) {
      p_pi[i] = host_print_values->pi[i];
    }
    return 0;
#endif // LB_GPU
  } else {
#ifdef LB
#ifndef LB_ADAPTIVE
    Lattice::index_t index;
    int node, grid[3], ind_shifted[3];
    double rho;
    double j[3];
    double pi[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    ind_shifted[0] = ind[0];
    ind_shifted[1] = ind[1];
    ind_shifted[2] = ind[2];
    node = lblattice.map_lattice_to_node(ind_shifted, grid);
    index = get_linear_index(ind_shifted[0], ind_shifted[1], ind_shifted[2],
                             lblattice.halo_grid);

    mpi_recv_fluid(node, index, &rho, j, pi);
    // unit conversion
    p_pi[0] = pi[0] / lbpar.tau / lbpar.tau / lbpar.agrid;
    p_pi[1] = pi[1] / lbpar.tau / lbpar.tau / lbpar.agrid;
    p_pi[2] = pi[2] / lbpar.tau / lbpar.tau / lbpar.agrid;
    p_pi[3] = pi[3] / lbpar.tau / lbpar.tau / lbpar.agrid;
    p_pi[4] = pi[4] / lbpar.tau / lbpar.tau / lbpar.agrid;
    p_pi[5] = pi[5] / lbpar.tau / lbpar.tau / lbpar.agrid;
#else  // !LB_ADAPTIVE
    double rho;
    double j[3];
    double pi[6];
    int64_t index = p4est_utils_cell_morton_idx(ind[0], ind[1], ind[2]);
    int proc = p4est_utils_idx_to_proc(forest_order::adaptive_LB, index);
    double h_max = 1.0 / double(1 << p4est_params.max_ref_level);
    mpi_recv_fluid(proc, index, &rho, j, pi);
    // unit conversion
    p_pi[0] = pi[0] / lbpar.tau / lbpar.tau / h_max;
    p_pi[1] = pi[1] / lbpar.tau / lbpar.tau / h_max;
    p_pi[2] = pi[2] / lbpar.tau / lbpar.tau / h_max;
    p_pi[3] = pi[3] / lbpar.tau / lbpar.tau / h_max;
    p_pi[4] = pi[4] / lbpar.tau / lbpar.tau / h_max;
    p_pi[5] = pi[5] / lbpar.tau / lbpar.tau / h_max;
#endif // !LB_ADAPTIVE
#endif // LB
  }
  return 0;
}

int lb_lbnode_get_boundary(const Vector3i &ind, int *p_boundary) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    unsigned int host_flag;
    int single_nodeindex = ind[0] + ind[1] * lbpar_gpu.dim_x +
                           ind[2] * lbpar_gpu.dim_x * lbpar_gpu.dim_y;
    lb_get_boundary_flag_GPU(single_nodeindex, &host_flag);
    p_boundary[0] = host_flag;
#endif // LB_GPU
  } else {
#ifdef LB
#ifndef LB_ADAPTIVE
    Lattice::index_t index;
    int node, grid[3], ind_shifted[3];

    ind_shifted[0] = ind[0];
    ind_shifted[1] = ind[1];
    ind_shifted[2] = ind[2];
    node = lblattice.map_lattice_to_node(ind_shifted, grid);
    index = get_linear_index(ind_shifted[0], ind_shifted[1], ind_shifted[2],
                             lblattice.halo_grid);

    mpi_recv_fluid_boundary_flag(node, index, p_boundary);
#else
    int64_t index = p4est_utils_cell_morton_idx(ind[0], ind[1], ind[2]);
    int proc = p4est_utils_idx_to_proc(forest_order::adaptive_LB, index);

    mpi_recv_fluid_boundary_flag(proc, index, p_boundary);
#endif // LB_ADAPTIVE
#endif // LB
  }
  return 0;
}

#ifdef LB_ADAPTIVE
void lb_local_fields_get_boundary_flag(uint64_t index, int *boundary) {
  p4est_locidx_t qid = p4est_utils_idx_to_qid(forest_order::adaptive_LB, index);
  P4EST_ASSERT(0 <= qid && qid < adapt_p4est->local_num_quadrants);

  auto quad = p8est_mesh_get_quadrant(adapt_p4est, adapt_mesh, qid);
  auto sid = adapt_virtual->quad_qreal_offset[qid];
  *boundary = lbadapt_local_data[quad->level][sid].lbfields.boundary;
}
#endif

#endif // defined (LB) || defined (LB_GPU)

int lb_lbnode_get_pop(const Vector3i &ind, double *p_pop) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    float population[19];

    // c is the LB_COMPONENT for SHANCHEN (not yet interfaced)
    int c = 0;
    lb_lbfluid_get_population(ind, population, c);

    for (int i = 0; i < LBQ; ++i)
      p_pop[i] = population[i];
#endif // LB_GPU
  } else {
#ifdef LB
#ifndef LB_ADAPTIVE
    Lattice::index_t index;
    int node, grid[3], ind_shifted[3];

    ind_shifted[0] = ind[0];
    ind_shifted[1] = ind[1];
    ind_shifted[2] = ind[2];
    node = lblattice.map_lattice_to_node(ind_shifted, grid);
    index = get_linear_index(ind_shifted[0], ind_shifted[1], ind_shifted[2],
                             lblattice.halo_grid);
    mpi_recv_fluid_populations(node, index, p_pop);
#endif // !LB_ADAPTIVE
#endif // LB
  }
  return 0;
}

int lb_lbnode_set_rho(const Vector3i &ind, double *p_rho) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    float host_rho[LB_COMPONENTS];
    int single_nodeindex = ind[0] + ind[1] * lbpar_gpu.dim_x +
                           ind[2] * lbpar_gpu.dim_x * lbpar_gpu.dim_y;
    int i;
    for (i = 0; i < LB_COMPONENTS; i++) {
      host_rho[i] = (float)p_rho[i];
    }
    lb_set_node_rho_GPU(single_nodeindex, host_rho);
#endif // LB_GPU
  } else {
#ifdef LB
#ifndef LB_ADAPTIVE
    Lattice::index_t index;
    int node, grid[3], ind_shifted[3];
    double rho;
    std::array<double, 3> j;
    std::array<double, 6> pi;

    ind_shifted[0] = ind[0];
    ind_shifted[1] = ind[1];
    ind_shifted[2] = ind[2];
    node = lblattice.map_lattice_to_node(ind_shifted, grid);
    index = get_linear_index(ind_shifted[0], ind_shifted[1], ind_shifted[2],
                             lblattice.halo_grid);

    mpi_recv_fluid(node, index, &rho, j.data(), pi.data());
    rho = (*p_rho) * lbpar.agrid * lbpar.agrid * lbpar.agrid;
    mpi_send_fluid(node, index, rho, j, pi);

//  lb_calc_average_rho();
//  lb_reinit_parameters();
#endif // LB_ADAPTIVE
#endif // LB
  }
  return 0;
}

int lb_lbnode_set_u(const Vector3i &ind, double *u) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    float host_velocity[3];
    host_velocity[0] = (float)u[0] * lbpar_gpu.tau / lbpar_gpu.agrid;
    host_velocity[1] = (float)u[1] * lbpar_gpu.tau / lbpar_gpu.agrid;
    host_velocity[2] = (float)u[2] * lbpar_gpu.tau / lbpar_gpu.agrid;
    int single_nodeindex = ind[0] + ind[1] * lbpar_gpu.dim_x +
                           ind[2] * lbpar_gpu.dim_x * lbpar_gpu.dim_y;
    lb_set_node_velocity_GPU(single_nodeindex, host_velocity);
#endif // LB_GPU
  } else {
#ifdef LB
    double rho;
    std::array<double, 3> j;
    std::array<double, 6> pi;
#ifndef LB_ADAPTIVE
    Lattice::index_t index;
    int node, grid[3], ind_shifted[3];

    ind_shifted[0] = ind[0];
    ind_shifted[1] = ind[1];
    ind_shifted[2] = ind[2];
    node = lblattice.map_lattice_to_node(ind_shifted, grid);
    index = get_linear_index(ind_shifted[0], ind_shifted[1], ind_shifted[2],
                             lblattice.halo_grid);

    /* transform to lattice units */

    mpi_recv_fluid(node, index, &rho, j.data(), pi.data());
    j[0] = rho * u[0] * lbpar.tau / lbpar.agrid;
    j[1] = rho * u[1] * lbpar.tau / lbpar.agrid;
    j[2] = rho * u[2] * lbpar.tau / lbpar.agrid;
    mpi_send_fluid(node, index, rho, j, pi);
#else
    int64_t index = p4est_utils_cell_morton_idx(ind[0], ind[1], ind[2]);
    int proc = p4est_utils_idx_to_proc(forest_order::adaptive_LB, index);
    lb_float h_max = p4est_params.h[p4est_params.max_ref_level];
    mpi_recv_fluid(proc, index, &rho, j.data(), pi.data());
    // unit conversion
    for (int i = 0; i < P8EST_DIM; ++i)
      j[i] = u[i] * lbpar.tau * rho / h_max;
    mpi_send_fluid(proc, index, rho, j, pi);
#endif // LB_ADAPTIVE
#endif // LB
  }
  return 0;
}

int lb_lbnode_set_pop(const Vector3i &ind, double *p_pop) {
  if (lattice_switch & LATTICE_LB_GPU) {
#ifdef LB_GPU
    float population[19];

    for (int i = 0; i < LBQ; ++i)
      population[i] = p_pop[i];

    // c is the LB_COMPONENT for SHANCHEN (not yet interfaced)
    int c = 0;
    lb_lbfluid_set_population(ind, population, c);
#endif // LB_GPU
  } else {
#ifdef LB
#ifndef LB_ADAPTIVE
    Lattice::index_t index;
    int node, grid[3], ind_shifted[3];

    ind_shifted[0] = ind[0];
    ind_shifted[1] = ind[1];
    ind_shifted[2] = ind[2];
    node = lblattice.map_lattice_to_node(ind_shifted, grid);
    index = get_linear_index(ind_shifted[0], ind_shifted[1], ind_shifted[2],
                             lblattice.halo_grid);
    mpi_send_fluid_populations(node, index, p_pop);
#endif // LB_ADAPTIVE
#endif // LB
  }
  return 0;
}

#ifdef LB
/********************** The Main LB Part *************************************/
/* Halo communication for push scheme */
#ifndef LB_ADAPTIVE
static void halo_push_communication(LB_Fluid &lbfluid) {
  Lattice::index_t index;
  int x, y, z, count;
  int rnode, snode;
  double *buffer = nullptr, *sbuf = nullptr, *rbuf = nullptr;
  MPI_Status status;

  int yperiod = lblattice.halo_grid[0];
  int zperiod = lblattice.halo_grid[0] * lblattice.halo_grid[1];

  /***************
   * X direction *
   ***************/
  count = 5 * lblattice.halo_grid[1] * lblattice.halo_grid[2];
  sbuf = (double *)Utils::malloc(count * sizeof(double));
  rbuf = (double *)Utils::malloc(count * sizeof(double));

  /* send to right, recv from left i = 1, 7, 9, 11, 13 */
  snode = node_neighbors[1];
  rnode = node_neighbors[0];

  buffer = sbuf;
  index = get_linear_index(lblattice.grid[0] + 1, 0, 0, lblattice.halo_grid);
  for (z = 0; z < lblattice.halo_grid[2]; z++) {
    for (y = 0; y < lblattice.halo_grid[1]; y++) {
      buffer[0] = lbfluid[1][index];
      buffer[1] = lbfluid[7][index];
      buffer[2] = lbfluid[9][index];
      buffer[3] = lbfluid[11][index];
      buffer[4] = lbfluid[13][index];
      buffer += 5;

      index += yperiod;
    }
  }

  if (node_grid[0] > 1) {
    MPI_Sendrecv(sbuf, count, MPI_DOUBLE, snode, REQ_HALO_SPREAD, rbuf, count,
                 MPI_DOUBLE, rnode, REQ_HALO_SPREAD, comm_cart, &status);
  } else {
    memmove(rbuf, sbuf, count * sizeof(double));
  }

  buffer = rbuf;
  index = get_linear_index(1, 0, 0, lblattice.halo_grid);
  for (z = 0; z < lblattice.halo_grid[2]; z++) {
    for (y = 0; y < lblattice.halo_grid[1]; y++) {
      lbfluid[1][index] = buffer[0];
      lbfluid[7][index] = buffer[1];
      lbfluid[9][index] = buffer[2];
      lbfluid[11][index] = buffer[3];
      lbfluid[13][index] = buffer[4];
      buffer += 5;

      index += yperiod;
    }
  }

  /* send to left, recv from right i = 2, 8, 10, 12, 14 */
  snode = node_neighbors[0];
  rnode = node_neighbors[1];

  buffer = sbuf;
  index = get_linear_index(0, 0, 0, lblattice.halo_grid);
  for (z = 0; z < lblattice.halo_grid[2]; z++) {
    for (y = 0; y < lblattice.halo_grid[1]; y++) {
      buffer[0] = lbfluid[2][index];
      buffer[1] = lbfluid[8][index];
      buffer[2] = lbfluid[10][index];
      buffer[3] = lbfluid[12][index];
      buffer[4] = lbfluid[14][index];
      buffer += 5;

      index += yperiod;
    }
  }

  if (node_grid[0] > 1) {
    MPI_Sendrecv(sbuf, count, MPI_DOUBLE, snode, REQ_HALO_SPREAD, rbuf, count,
                 MPI_DOUBLE, rnode, REQ_HALO_SPREAD, comm_cart, &status);
  } else {
    memmove(rbuf, sbuf, count * sizeof(double));
  }

  buffer = rbuf;
  index = get_linear_index(lblattice.grid[0], 0, 0, lblattice.halo_grid);
  for (z = 0; z < lblattice.halo_grid[2]; z++) {
    for (y = 0; y < lblattice.halo_grid[1]; y++) {
      lbfluid[2][index] = buffer[0];
      lbfluid[8][index] = buffer[1];
      lbfluid[10][index] = buffer[2];
      lbfluid[12][index] = buffer[3];
      lbfluid[14][index] = buffer[4];
      buffer += 5;

      index += yperiod;
    }
  }

  /***************
   * Y direction *
   ***************/
  count = 5 * lblattice.halo_grid[0] * lblattice.halo_grid[2];
  sbuf = Utils::realloc(sbuf, count * sizeof(double));
  rbuf = Utils::realloc(rbuf, count * sizeof(double));

  /* send to right, recv from left i = 3, 7, 10, 15, 17 */
  snode = node_neighbors[3];
  rnode = node_neighbors[2];

  buffer = sbuf;
  index = get_linear_index(0, lblattice.grid[1] + 1, 0, lblattice.halo_grid);
  for (z = 0; z < lblattice.halo_grid[2]; z++) {
    for (x = 0; x < lblattice.halo_grid[0]; x++) {
      buffer[0] = lbfluid[3][index];
      buffer[1] = lbfluid[7][index];
      buffer[2] = lbfluid[10][index];
      buffer[3] = lbfluid[15][index];
      buffer[4] = lbfluid[17][index];
      buffer += 5;

      ++index;
    }
    index += zperiod - lblattice.halo_grid[0];
  }

  if (node_grid[1] > 1) {
    MPI_Sendrecv(sbuf, count, MPI_DOUBLE, snode, REQ_HALO_SPREAD, rbuf, count,
                 MPI_DOUBLE, rnode, REQ_HALO_SPREAD, comm_cart, &status);
  } else {
    memmove(rbuf, sbuf, count * sizeof(double));
  }

  buffer = rbuf;
  index = get_linear_index(0, 1, 0, lblattice.halo_grid);
  for (z = 0; z < lblattice.halo_grid[2]; z++) {
    for (x = 0; x < lblattice.halo_grid[0]; x++) {
      lbfluid[3][index] = buffer[0];
      lbfluid[7][index] = buffer[1];
      lbfluid[10][index] = buffer[2];
      lbfluid[15][index] = buffer[3];
      lbfluid[17][index] = buffer[4];
      buffer += 5;

      ++index;
    }
    index += zperiod - lblattice.halo_grid[0];
  }

  /* send to left, recv from right i = 4, 8, 9, 16, 18 */
  snode = node_neighbors[2];
  rnode = node_neighbors[3];

  buffer = sbuf;
  index = get_linear_index(0, 0, 0, lblattice.halo_grid);
  for (z = 0; z < lblattice.halo_grid[2]; z++) {
    for (x = 0; x < lblattice.halo_grid[0]; x++) {
      buffer[0] = lbfluid[4][index];
      buffer[1] = lbfluid[8][index];
      buffer[2] = lbfluid[9][index];
      buffer[3] = lbfluid[16][index];
      buffer[4] = lbfluid[18][index];
      buffer += 5;

      ++index;
    }
    index += zperiod - lblattice.halo_grid[0];
  }

  if (node_grid[1] > 1) {
    MPI_Sendrecv(sbuf, count, MPI_DOUBLE, snode, REQ_HALO_SPREAD, rbuf, count,
                 MPI_DOUBLE, rnode, REQ_HALO_SPREAD, comm_cart, &status);
  } else {
    memmove(rbuf, sbuf, count * sizeof(double));
  }

  buffer = rbuf;
  index = get_linear_index(0, lblattice.grid[1], 0, lblattice.halo_grid);
  for (z = 0; z < lblattice.halo_grid[2]; z++) {
    for (x = 0; x < lblattice.halo_grid[0]; x++) {
      lbfluid[4][index] = buffer[0];
      lbfluid[8][index] = buffer[1];
      lbfluid[9][index] = buffer[2];
      lbfluid[16][index] = buffer[3];
      lbfluid[18][index] = buffer[4];
      buffer += 5;

      ++index;
    }
    index += zperiod - lblattice.halo_grid[0];
  }

  /***************
   * Z direction *
   ***************/
  count = 5 * lblattice.halo_grid[0] * lblattice.halo_grid[1];
  sbuf = Utils::realloc(sbuf, count * sizeof(double));
  rbuf = Utils::realloc(rbuf, count * sizeof(double));

  /* send to right, recv from left i = 5, 11, 14, 15, 18 */
  snode = node_neighbors[5];
  rnode = node_neighbors[4];

  buffer = sbuf;
  index = get_linear_index(0, 0, lblattice.grid[2] + 1, lblattice.halo_grid);
  for (y = 0; y < lblattice.halo_grid[1]; y++) {
    for (x = 0; x < lblattice.halo_grid[0]; x++) {
      buffer[0] = lbfluid[5][index];
      buffer[1] = lbfluid[11][index];
      buffer[2] = lbfluid[14][index];
      buffer[3] = lbfluid[15][index];
      buffer[4] = lbfluid[18][index];
      buffer += 5;

      ++index;
    }
  }

  if (node_grid[2] > 1) {
    MPI_Sendrecv(sbuf, count, MPI_DOUBLE, snode, REQ_HALO_SPREAD, rbuf, count,
                 MPI_DOUBLE, rnode, REQ_HALO_SPREAD, comm_cart, &status);
  } else {
    memmove(rbuf, sbuf, count * sizeof(double));
  }

  buffer = rbuf;
  index = get_linear_index(0, 0, 1, lblattice.halo_grid);
  for (y = 0; y < lblattice.halo_grid[1]; y++) {
    for (x = 0; x < lblattice.halo_grid[0]; x++) {
      lbfluid[5][index] = buffer[0];
      lbfluid[11][index] = buffer[1];
      lbfluid[14][index] = buffer[2];
      lbfluid[15][index] = buffer[3];
      lbfluid[18][index] = buffer[4];
      buffer += 5;

      ++index;
    }
  }

  /* send to left, recv from right i = 6, 12, 13, 16, 17 */
  snode = node_neighbors[4];
  rnode = node_neighbors[5];

  buffer = sbuf;
  index = get_linear_index(0, 0, 0, lblattice.halo_grid);
  for (y = 0; y < lblattice.halo_grid[1]; y++) {
    for (x = 0; x < lblattice.halo_grid[0]; x++) {
      buffer[0] = lbfluid[6][index];
      buffer[1] = lbfluid[12][index];
      buffer[2] = lbfluid[13][index];
      buffer[3] = lbfluid[16][index];
      buffer[4] = lbfluid[17][index];
      buffer += 5;

      ++index;
    }
  }

  if (node_grid[2] > 1) {
    MPI_Sendrecv(sbuf, count, MPI_DOUBLE, snode, REQ_HALO_SPREAD, rbuf, count,
                 MPI_DOUBLE, rnode, REQ_HALO_SPREAD, comm_cart, &status);
  } else {
    memmove(rbuf, sbuf, count * sizeof(double));
  }

  buffer = rbuf;
  index = get_linear_index(0, 0, lblattice.grid[2], lblattice.halo_grid);
  for (y = 0; y < lblattice.halo_grid[1]; y++) {
    for (x = 0; x < lblattice.halo_grid[0]; x++) {
      lbfluid[6][index] = buffer[0];
      lbfluid[12][index] = buffer[1];
      lbfluid[13][index] = buffer[2];
      lbfluid[16][index] = buffer[3];
      lbfluid[17][index] = buffer[4];
      buffer += 5;

      ++index;
    }
  }

  free(rbuf);
  free(sbuf);
}
#endif // LB_ADAPTIVE

/***********************************************************************/

/** Performs basic sanity checks. */
int lb_sanity_checks() {
  // char *errtext;
  int ret = 0;

  if (lbpar.agrid <= 0.0) {
    runtimeErrorMsg() << "Lattice Boltzmann agrid not set";
    ret = 1;
  }
  if (lbpar.tau <= 0.0) {
    runtimeErrorMsg() << "Lattice Boltzmann time step not set";
    ret = 1;
  }
  if (lbpar.rho <= 0.0) {
    runtimeErrorMsg() << "Lattice Boltzmann fluid density not set";
    ret = 1;
  }
  if (lbpar.viscosity <= 0.0) {
    runtimeErrorMsg() << "Lattice Boltzmann fluid viscosity not set";
    ret = 1;
  }
#ifndef LB_ADAPTIVE
  if (cell_structure.type != CELL_STRUCTURE_DOMDEC) {
    runtimeErrorMsg() << "LB requires domain-decomposition cellsystem";
    ret = -1;
  }
#else
  if (!local_cells.particles().empty() &&
      cell_structure.type != CELL_STRUCTURE_P4EST) {
    runtimeErrorMsg() << "Adaptive LB requires p4est cellsystem";
    ret = -1;
  }
#endif
#ifndef LB_ADAPTIVE
  if (skin == 0.0) {
    runtimeErrorMsg() << "LB requires a positive skin";
    ret = 1;
  }
#endif
  if (cell_structure.use_verlet_list && skin >= lbpar.agrid / 2.0) {
    runtimeErrorMsg() << "LB requires either no Verlet lists or that the skin "
                         "of the verlet list to be less than half of "
                         "lattice-Boltzmann grid spacing";
    ret = -1;
  }
  return ret;
}

/***********************************************************************/

/** (Re-)allocate memory for the fluid and initialize pointers. */
static void lb_realloc_fluid() {
#ifndef LB_ADAPTIVE
  LB_TRACE(printf("reallocating fluid\n"));
  const std::array<int, 2> size = {
      {lbmodel.n_veloc, lblattice.halo_grid_volume}};

  lbfluid_a.resize(size);
  lbfluid_b.resize(size);

  using Utils::Span;
  for (int i = 0; i < size[0]; i++) {
    lbfluid[i] = Span<double>(lbfluid_a[i].origin(), size[1]);
    lbfluid_post[i] = Span<double>(lbfluid_b[i].origin(), size[1]);
  }

  lbfields.resize(lblattice.halo_grid_volume);
#endif // LB_ADAPTIVE
}

/** Sets up the structures for exchange of the halo regions.
 *  See also \ref halo.cpp */
static void lb_prepare_communication() {
#ifndef LB_ADAPTIVE
  int i;
  HaloCommunicator comm = {0, nullptr};

  /* since the data layout is a structure of arrays, we have to
   * generate a communication for this structure: first we generate
   * the communication for one of the arrays (the 0-th velocity
   * population), then we replicate this communication for the other
   * velocity indices by constructing appropriate vector
   * datatypes */

  /* prepare the communication for a single velocity */
  prepare_halo_communication(&comm, &lblattice, FIELDTYPE_DOUBLE, MPI_DOUBLE);

  update_halo_comm.num = comm.num;
  update_halo_comm.halo_info =
      Utils::realloc(update_halo_comm.halo_info, comm.num * sizeof(HaloInfo));

  /* replicate the halo structure */
  for (i = 0; i < comm.num; i++) {
    HaloInfo *hinfo = &(update_halo_comm.halo_info[i]);

    hinfo->source_node = comm.halo_info[i].source_node;
    hinfo->dest_node = comm.halo_info[i].dest_node;
    hinfo->s_offset = comm.halo_info[i].s_offset;
    hinfo->r_offset = comm.halo_info[i].r_offset;
    hinfo->type = comm.halo_info[i].type;

    /* generate the vector datatype for the structure of lattices we
     * have to use hvector here because the extent of the subtypes
     * does not span the full lattice and hence we cannot get the
     * correct vskip out of them */

    MPI_Aint lower;
    MPI_Aint extent;
    MPI_Type_get_extent(MPI_DOUBLE, &lower, &extent);
    MPI_Type_create_hvector(lbmodel.n_veloc, 1,
                            lblattice.halo_grid_volume * extent,
                            comm.halo_info[i].datatype, &hinfo->datatype);
    MPI_Type_commit(&hinfo->datatype);

    halo_create_field_hvector(lbmodel.n_veloc, 1,
                              lblattice.halo_grid_volume * sizeof(double),
                              comm.halo_info[i].fieldtype, &hinfo->fieldtype);
  }

  release_halo_communication(&comm);
#else
  runtimeErrorMsg() << __FUNCTION__ << " not implemented with LB_ADAPTIVE flag";
#endif // LB_ADAPTIVE
}

/** (Re-)initializes the fluid. */
void lb_reinit_parameters() {
#ifdef LB_ADAPTIVE
  lbadapt_reinit_parameters();
#else  // LB_ADAPTIVE
  int i;
  if (lbpar.viscosity > 0.0) {
    /* Eq. (80) Duenweg, Schiller, Ladd, PRE 76(3):036704 (2007). */
    // unit conversion: viscosity
    lbpar.gamma_shear = 1. - 2. / (6. * lbpar.viscosity * lbpar.tau /
                                       (lbpar.agrid * lbpar.agrid) +
                                   1.);
  }

  if (lbpar.bulk_viscosity > 0.0) {
    /* Eq. (81) Duenweg, Schiller, Ladd, PRE 76(3):036704 (2007). */
    // unit conversion: viscosity
    lbpar.gamma_bulk = 1. - 2. / (9. * lbpar.bulk_viscosity * lbpar.tau /
                                      (lbpar.agrid * lbpar.agrid) +
                                  1.);
  }

  if (lbpar.is_TRT) {
    lbpar.gamma_bulk = lbpar.gamma_shear;
    lbpar.gamma_even = lbpar.gamma_shear;
    lbpar.gamma_odd =
        -(7.0 * lbpar.gamma_even + 1.0) / (lbpar.gamma_even + 7.0);
    // gamma_odd = lbpar.gamma_shear; //uncomment for BGK
  }

  // lbpar.gamma_shear = 0.0; //uncomment for special case of BGK
  // lbpar.gamma_bulk = 0.0;
  // gamma_odd = 0.0;
  // gamma_even = 0.0;

  if (temperature > 0.0) {
    /* fluctuating hydrodynamics ? */
    lbpar.fluct = 1;

    /* Eq. (51) Duenweg, Schiller, Ladd, PRE 76(3):036704 (2007).
     * Note that the modes are not normalized as in the paper here! */
    double mu = temperature / lbmodel.c_sound_sq * lbpar.tau * lbpar.tau /
                (lbpar.agrid * lbpar.agrid);
    // mu *= agrid*agrid*agrid;  // Marcello's conjecture

    for (i = 0; i < 4; i++)
      lbpar.phi[i] = 0.0;

    lbpar.phi[4] =
        sqrt(mu * lbmodel.e[19][4] *
             (1. - Utils::sqr(lbpar.gamma_bulk))); // Utils::sqr(x) == x*x
    for (i = 5; i < 10; i++)
      lbpar.phi[i] =
          sqrt(mu * lbmodel.e[19][i] * (1. - Utils::sqr(lbpar.gamma_shear)));
    for (i = 10; i < 16; i++)
      lbpar.phi[i] =
          sqrt(mu * lbmodel.e[19][i] * (1 - Utils::sqr(lbpar.gamma_odd)));
    for (i = 16; i < 19; i++)
      lbpar.phi[i] =
          sqrt(mu * lbmodel.e[19][i] * (1 - Utils::sqr(lbpar.gamma_even)));

    /* lb_coupl_pref is stored in MD units (force)
     * Eq. (16) Ahlrichs and Duenweg, JCP 111(17):8225 (1999).
     * The factor 12 comes from the fact that we use random numbers
     * from -0.5 to 0.5 (equally distributed) which have variance 1/12.
     * time_step comes from the discretization.
     */
    lb_coupl_pref = sqrt(12. * 2. * lbpar.friction * temperature / time_step);
    lb_coupl_pref2 = sqrt(2. * lbpar.friction * temperature / time_step);
    LB_TRACE(fprintf(
        stderr,
        "%d: lbpar.gamma_shear=%lf lbpar.gamma_bulk=%lf shear_fluct=%lf "
        "bulk_fluct=%lf mu=%lf, bulkvisc=%lf, visc=%lf\n",
        this_node, lbpar.gamma_shear, lbpar.gamma_bulk, lbpar.phi[9],
        lbpar.phi[4], mu, lbpar.bulk_viscosity, lbpar.viscosity));
  } else {
    /* no fluctuations at zero temperature */
    lbpar.fluct = 0;
    for (i = 0; i < lbmodel.n_veloc; i++)
      lbpar.phi[i] = 0.0;
    lb_coupl_pref = 0.0;
    lb_coupl_pref2 = 0.0;
  }
#endif // LB_ADAPTIVE
}

/** (Re-)initializes the fluid according to the given value of rho. */
void lb_reinit_fluid() {
#ifdef LB_ADAPTIVE
  lbadapt_reinit_fluid_per_cell();
#else // LB_ADAPTIVE
  std::fill(lbfields.begin(), lbfields.end(), LB_FluidNode());
  /* default values for fields in lattice units */
  /* here the conversion to lb units is performed */
  double rho = lbpar.rho * lbpar.agrid * lbpar.agrid * lbpar.agrid;
  std::array<double, 3> j = {{0., 0., 0.}};
  std::array<double, 6> pi = {{0., 0., 0., 0., 0., 0.}};

  LB_TRACE(fprintf(stderr,
                   "Initialising the fluid with equilibrium populations\n"););

  for (Lattice::index_t index = 0; index < lblattice.halo_grid_volume;
       ++index) {
    // calculate equilibrium distribution
    lb_calc_n_from_rho_j_pi(index, rho, j, pi);

#ifdef LB_BOUNDARIES
    lbfields[index].boundary = 0;
#endif // LB_BOUNDARIES
  }
#endif // LB_ADAPTIVE

  lbpar.resend_halo = 0;
#ifdef LB_BOUNDARIES
  LBBoundaries::lb_init_boundaries();
#endif // LB_BOUNDARIES
}

/** Performs a full initialization of
 *  the Lattice Boltzmann system. All derived parameters
 *  and the fluid are reset to their default values. */
void lb_init() {
  LB_TRACE(printf("Begin initialzing fluid on CPU\n"));

/* allocate regular grid */
#ifndef LB_ADAPTIVE
  if (lbpar.agrid <= 0.0) {
    runtimeErrorMsg()
        << "Lattice Boltzmann agrid not set when initializing fluid";
  }

  if (check_runtime_errors())
    return;

  double temp_agrid[3];
  double temp_offset[3];
  for (int i = 0; i < 3; i++) {
    temp_agrid[i] = lbpar.agrid;
    temp_offset[i] = 0.5;
  }

  /* initialize the local lattice domain */
  lblattice.init(temp_agrid, temp_offset, 1, 0);

  if (check_runtime_errors())
    return;

  /* allocate memory for data structures */
  lb_realloc_fluid();

  /* prepare the halo communication */
  lb_prepare_communication();
#else  // !LB_ADAPTIVE
  lbadapt_init();
#endif // !LB_ADAPTIVE

  /* initialize derived parameters */
  lb_reinit_parameters();

  /* setup the initial particle velocity distribution */
  lb_reinit_fluid();

  LB_TRACE(printf("Initialzing fluid on CPU successful\n"));
}

/** Release fluid and communication. */
#ifndef LB_ADAPTIVE
void lb_release() { release_halo_communication(&update_halo_comm); }
#endif

/***********************************************************************/
/** \name Mapping between hydrodynamic fields and particle populations */
/***********************************************************************/
/*@{*/
void lb_calc_n_from_rho_j_pi(const Lattice::index_t index, const double rho,
                             const std::array<double, 3> &j,
                             const std::array<double, 6> &pi) {
#ifndef LB_ADAPTIVE
  int i;
  double local_rho, local_j[3], local_pi[6], trace;
  const double avg_rho = lbpar.rho * lbpar.agrid * lbpar.agrid * lbpar.agrid;

  local_rho = rho;

  local_j[0] = j[0];
  local_j[1] = j[1];
  local_j[2] = j[2];

  for (i = 0; i < 6; i++)
    local_pi[i] = pi[i];

  trace = local_pi[0] + local_pi[2] + local_pi[5];

  double rho_times_coeff;
  double tmp1, tmp2;

  /* update the q=0 sublattice */
  lbfluid[0][index] = 1. / 3. * (local_rho - avg_rho) - 1. / 2. * trace;

  /* update the q=1 sublattice */
  rho_times_coeff = 1. / 18. * (local_rho - avg_rho);

  lbfluid[1][index] = rho_times_coeff + 1. / 6. * local_j[0] +
                      1. / 4. * local_pi[0] - 1. / 12. * trace;
  lbfluid[2][index] = rho_times_coeff - 1. / 6. * local_j[0] +
                      1. / 4. * local_pi[0] - 1. / 12. * trace;
  lbfluid[3][index] = rho_times_coeff + 1. / 6. * local_j[1] +
                      1. / 4. * local_pi[2] - 1. / 12. * trace;
  lbfluid[4][index] = rho_times_coeff - 1. / 6. * local_j[1] +
                      1. / 4. * local_pi[2] - 1. / 12. * trace;
  lbfluid[5][index] = rho_times_coeff + 1. / 6. * local_j[2] +
                      1. / 4. * local_pi[5] - 1. / 12. * trace;
  lbfluid[6][index] = rho_times_coeff - 1. / 6. * local_j[2] +
                      1. / 4. * local_pi[5] - 1. / 12. * trace;

  /* update the q=2 sublattice */
  rho_times_coeff = 1. / 36. * (local_rho - avg_rho);

  tmp1 = local_pi[0] + local_pi[2];
  tmp2 = 2.0 * local_pi[1];

  lbfluid[7][index] = rho_times_coeff + 1. / 12. * (local_j[0] + local_j[1]) +
                      1. / 8. * (tmp1 + tmp2) - 1. / 24. * trace;
  lbfluid[8][index] = rho_times_coeff - 1. / 12. * (local_j[0] + local_j[1]) +
                      1. / 8. * (tmp1 + tmp2) - 1. / 24. * trace;
  lbfluid[9][index] = rho_times_coeff + 1. / 12. * (local_j[0] - local_j[1]) +
                      1. / 8. * (tmp1 - tmp2) - 1. / 24. * trace;
  lbfluid[10][index] = rho_times_coeff - 1. / 12. * (local_j[0] - local_j[1]) +
                       1. / 8. * (tmp1 - tmp2) - 1. / 24. * trace;

  tmp1 = local_pi[0] + local_pi[5];
  tmp2 = 2.0 * local_pi[3];

  lbfluid[11][index] = rho_times_coeff + 1. / 12. * (local_j[0] + local_j[2]) +
                       1. / 8. * (tmp1 + tmp2) - 1. / 24. * trace;
  lbfluid[12][index] = rho_times_coeff - 1. / 12. * (local_j[0] + local_j[2]) +
                       1. / 8. * (tmp1 + tmp2) - 1. / 24. * trace;
  lbfluid[13][index] = rho_times_coeff + 1. / 12. * (local_j[0] - local_j[2]) +
                       1. / 8. * (tmp1 - tmp2) - 1. / 24. * trace;
  lbfluid[14][index] = rho_times_coeff - 1. / 12. * (local_j[0] - local_j[2]) +
                       1. / 8. * (tmp1 - tmp2) - 1. / 24. * trace;

  tmp1 = local_pi[2] + local_pi[5];
  tmp2 = 2.0 * local_pi[4];

  lbfluid[15][index] = rho_times_coeff + 1. / 12. * (local_j[1] + local_j[2]) +
                       1. / 8. * (tmp1 + tmp2) - 1. / 24. * trace;
  lbfluid[16][index] = rho_times_coeff - 1. / 12. * (local_j[1] + local_j[2]) +
                       1. / 8. * (tmp1 + tmp2) - 1. / 24. * trace;
  lbfluid[17][index] = rho_times_coeff + 1. / 12. * (local_j[1] - local_j[2]) +
                       1. / 8. * (tmp1 - tmp2) - 1. / 24. * trace;
  lbfluid[18][index] = rho_times_coeff - 1. / 12. * (local_j[1] - local_j[2]) +
                       1. / 8. * (tmp1 - tmp2) - 1. / 24. * trace;
#else
  // get qid
  int qid = p4est_utils_idx_to_qid(forest_order::adaptive_LB, index);
  P4EST_ASSERT(0 <= qid && qid < adapt_p4est->local_num_quadrants);
  p8est_quadrant_t *q = p8est_mesh_get_quadrant(adapt_p4est, adapt_mesh, qid);
  lbadapt_payload_t *data =
      &lbadapt_local_data[q->level].at(adapt_virtual->quad_qreal_offset[qid]);
  lb_float j_cast[3];
  for (int i = 0; i < P8EST_DIM; ++i)
    j_cast[i] = static_cast<lb_float>(j[i]);
  lbadapt_calc_n_from_rho_j_pi(data->lbfluid, rho, j_cast, pi,
                               p4est_params.h[q->level]);
#endif // !LB_ADAPTIVE
}

/*@}*/

/** Calculation of hydrodynamic modes */
void lb_calc_modes(Lattice::index_t index, double *mode) {
#ifndef LB_ADAPTIVE
  double n0, n1p, n1m, n2p, n2m, n3p, n3m, n4p, n4m, n5p, n5m, n6p, n6m, n7p,
      n7m, n8p, n8m, n9p, n9m;

  n0 = lbfluid[0][index];
  n1p = lbfluid[1][index] + lbfluid[2][index];
  n1m = lbfluid[1][index] - lbfluid[2][index];
  n2p = lbfluid[3][index] + lbfluid[4][index];
  n2m = lbfluid[3][index] - lbfluid[4][index];
  n3p = lbfluid[5][index] + lbfluid[6][index];
  n3m = lbfluid[5][index] - lbfluid[6][index];
  n4p = lbfluid[7][index] + lbfluid[8][index];
  n4m = lbfluid[7][index] - lbfluid[8][index];
  n5p = lbfluid[9][index] + lbfluid[10][index];
  n5m = lbfluid[9][index] - lbfluid[10][index];
  n6p = lbfluid[11][index] + lbfluid[12][index];
  n6m = lbfluid[11][index] - lbfluid[12][index];
  n7p = lbfluid[13][index] + lbfluid[14][index];
  n7m = lbfluid[13][index] - lbfluid[14][index];
  n8p = lbfluid[15][index] + lbfluid[16][index];
  n8m = lbfluid[15][index] - lbfluid[16][index];
  n9p = lbfluid[17][index] + lbfluid[18][index];
  n9m = lbfluid[17][index] - lbfluid[18][index];

  /* mass mode */
  mode[0] = n0 + n1p + n2p + n3p + n4p + n5p + n6p + n7p + n8p + n9p;

  /* momentum modes */
  mode[1] = n1m + n4m + n5m + n6m + n7m;
  mode[2] = n2m + n4m - n5m + n8m + n9m;
  mode[3] = n3m + n6m - n7m + n8m - n9m;

  /* stress modes */
  mode[4] = -n0 + n4p + n5p + n6p + n7p + n8p + n9p;
  mode[5] = n1p - n2p + n6p + n7p - n8p - n9p;
  mode[6] = n1p + n2p - n6p - n7p - n8p - n9p - 2. * (n3p - n4p - n5p);
  mode[7] = n4p - n5p;
  mode[8] = n6p - n7p;
  mode[9] = n8p - n9p;

  /* kinetic modes */
  mode[10] = -2. * n1m + n4m + n5m + n6m + n7m;
  mode[11] = -2. * n2m + n4m - n5m + n8m + n9m;
  mode[12] = -2. * n3m + n6m - n7m + n8m - n9m;
  mode[13] = n4m + n5m - n6m - n7m;
  mode[14] = n4m - n5m - n8m - n9m;
  mode[15] = n6m - n7m - n8m + n9m;
  mode[16] = n0 + n4p + n5p + n6p + n7p + n8p + n9p - 2. * (n1p + n2p + n3p);
  mode[17] = -n1p + n2p + n6p + n7p - n8p - n9p;
  mode[18] = -n1p - n2p - n6p - n7p - n8p - n9p + 2. * (n3p + n4p + n5p);
#else  // !LB_ADAPTIVE
  runtimeErrorMsg() << __FUNCTION__ << " not implemented with LB_ADAPTIVE flag";
#endif // !LB_ADAPTIVE
}

inline void lb_relax_modes(Lattice::index_t index, double *mode) {
#ifndef LB_ADAPTIVE
  double rho, j[3], pi_eq[6];

  /* re-construct the real density
   * remember that the populations are stored as differences to their
   * equilibrium value */
  rho = mode[0] + lbpar.rho * lbpar.agrid * lbpar.agrid * lbpar.agrid;

  j[0] = mode[1] + 0.5 * lbfields[index].force_density[0];
  j[1] = mode[2] + 0.5 * lbfields[index].force_density[1];
  j[2] = mode[3] + 0.5 * lbfields[index].force_density[2];

  /* equilibrium part of the stress modes */
  pi_eq[0] = scalar(j, j) / rho;
  pi_eq[1] = (Utils::sqr(j[0]) - Utils::sqr(j[1])) / rho;
  pi_eq[2] = (scalar(j, j) - 3.0 * Utils::sqr(j[2])) / rho;
  pi_eq[3] = j[0] * j[1] / rho;
  pi_eq[4] = j[0] * j[2] / rho;
  pi_eq[5] = j[1] * j[2] / rho;

  /* relax the stress modes */
  mode[4] = pi_eq[0] + lbpar.gamma_bulk * (mode[4] - pi_eq[0]);
  mode[5] = pi_eq[1] + lbpar.gamma_shear * (mode[5] - pi_eq[1]);
  mode[6] = pi_eq[2] + lbpar.gamma_shear * (mode[6] - pi_eq[2]);
  mode[7] = pi_eq[3] + lbpar.gamma_shear * (mode[7] - pi_eq[3]);
  mode[8] = pi_eq[4] + lbpar.gamma_shear * (mode[8] - pi_eq[4]);
  mode[9] = pi_eq[5] + lbpar.gamma_shear * (mode[9] - pi_eq[5]);

  /* relax the ghost modes (project them out) */
  /* ghost modes have no equilibrium part due to orthogonality */
  mode[10] = lbpar.gamma_odd * mode[10];
  mode[11] = lbpar.gamma_odd * mode[11];
  mode[12] = lbpar.gamma_odd * mode[12];
  mode[13] = lbpar.gamma_odd * mode[13];
  mode[14] = lbpar.gamma_odd * mode[14];
  mode[15] = lbpar.gamma_odd * mode[15];
  mode[16] = lbpar.gamma_even * mode[16];
  mode[17] = lbpar.gamma_even * mode[17];
  mode[18] = lbpar.gamma_even * mode[18];
#else
  runtimeErrorMsg() << __FUNCTION__ << " not implemented with LB_ADAPTIVE flag";
#endif // LB_ADAPTIVE
}

inline void lb_thermalize_modes(Lattice::index_t index, double *mode) {
  const double rootrho = std::sqrt(
      std::fabs(mode[0] + lbpar.rho * lbpar.agrid * lbpar.agrid * lbpar.agrid));
#ifdef GAUSSRANDOM
  constexpr double variance = 1.0;
  auto rng = []() -> double { return gaussian_random(); };
#elif defined(GAUSSRANDOMCUT)
  constexpr double variance = 1.0;
  auto rng = []() -> double { return gaussian_random_cut(); };
#elif defined(FLATNOISE)
  constexpr double variance = 1. / 12.0;
  auto rng = []() -> double { return d_random() - 0.5; };
#else // GAUSSRANDOM
#error No noise type defined for the CPU LB
#endif // GAUSSRANDOM

  auto const pref = std::sqrt(1. / variance) * rootrho;

  /* stress modes */
  mode[4] += pref * lbpar.phi[4] * rng();
  mode[5] += pref * lbpar.phi[5] * rng();
  mode[6] += pref * lbpar.phi[6] * rng();
  mode[7] += pref * lbpar.phi[7] * rng();
  mode[8] += pref * lbpar.phi[8] * rng();
  mode[9] += pref * lbpar.phi[9] * rng();

  /* ghost modes */
  mode[10] += pref * lbpar.phi[10] * rng();
  mode[11] += pref * lbpar.phi[11] * rng();
  mode[12] += pref * lbpar.phi[12] * rng();
  mode[13] += pref * lbpar.phi[13] * rng();
  mode[14] += pref * lbpar.phi[14] * rng();
  mode[15] += pref * lbpar.phi[15] * rng();
  mode[16] += pref * lbpar.phi[16] * rng();
  mode[17] += pref * lbpar.phi[17] * rng();
  mode[18] += pref * lbpar.phi[18] * rng();

#ifdef ADDITIONAL_CHECKS
  rancounter += 15;
#endif // ADDITIONAL_CHECKS
}

inline void lb_apply_forces(Lattice::index_t index, double *mode) {
#ifndef LB_ADAPTIVE
  double rho, *f, u[3], C[6];

  f = lbfields[index].force_density;

  rho = mode[0] + lbpar.rho * lbpar.agrid * lbpar.agrid * lbpar.agrid;

  /* hydrodynamic momentum density is redefined when external forces present */
  u[0] = (mode[1] + 0.5 * f[0]) / rho;
  u[1] = (mode[2] + 0.5 * f[1]) / rho;
  u[2] = (mode[3] + 0.5 * f[2]) / rho;

  C[0] = (1. + lbpar.gamma_bulk) * u[0] * f[0] +
         1. / 3. * (lbpar.gamma_bulk - lbpar.gamma_shear) * scalar(u, f);
  C[2] = (1. + lbpar.gamma_bulk) * u[1] * f[1] +
         1. / 3. * (lbpar.gamma_bulk - lbpar.gamma_shear) * scalar(u, f);
  C[5] = (1. + lbpar.gamma_bulk) * u[2] * f[2] +
         1. / 3. * (lbpar.gamma_bulk - lbpar.gamma_shear) * scalar(u, f);
  C[1] = 1. / 2. * (1. + lbpar.gamma_shear) * (u[0] * f[1] + u[1] * f[0]);
  C[3] = 1. / 2. * (1. + lbpar.gamma_shear) * (u[0] * f[2] + u[2] * f[0]);
  C[4] = 1. / 2. * (1. + lbpar.gamma_shear) * (u[1] * f[2] + u[2] * f[1]);

  /* update momentum modes */
  mode[1] += f[0];
  mode[2] += f[1];
  mode[3] += f[2];

  /* update stress modes */
  mode[4] += C[0] + C[2] + C[5];
  mode[5] += C[0] - C[2];
  mode[6] += C[0] + C[2] - 2. * C[5];
  mode[7] += C[1];
  mode[8] += C[3];
  mode[9] += C[4];
#else
  runtimeErrorMsg() << __FUNCTION__ << " not implemented with LB_ADAPTIVE flag";
#endif // LB_ADAPTIVE
}

inline void lb_reset_force_densities(Lattice::index_t index) {
#ifndef LB_ADAPTIVE
  /* reset force */
  // unit conversion: force density
  lbfields[index].force_density[0] = lbpar.ext_force_density[0] * lbpar.agrid *
                                     lbpar.agrid * lbpar.tau * lbpar.tau;
  lbfields[index].force_density[1] = lbpar.ext_force_density[1] * lbpar.agrid *
                                     lbpar.agrid * lbpar.tau * lbpar.tau;
  lbfields[index].force_density[2] = lbpar.ext_force_density[2] * lbpar.agrid *
                                     lbpar.agrid * lbpar.tau * lbpar.tau;
#else
  runtimeErrorMsg() << __FUNCTION__ << " not implemented with LB_ADAPTIVE flag";
#endif // LB_ADAPTIVE
}

#ifndef LB_ADAPTIVE
inline void lb_calc_n_from_modes_push(LB_Fluid &lbfluid, Lattice::index_t index,
                                      double *m) {
  int yperiod = lblattice.halo_grid[0];
  int zperiod = lblattice.halo_grid[0] * lblattice.halo_grid[1];
  Lattice::index_t next[19];
  next[0] = index;
  next[1] = index + 1;
  next[2] = index - 1;
  next[3] = index + yperiod;
  next[4] = index - yperiod;
  next[5] = index + zperiod;
  next[6] = index - zperiod;
  next[7] = index + (1 + yperiod);
  next[8] = index - (1 + yperiod);
  next[9] = index + (1 - yperiod);
  next[10] = index - (1 - yperiod);
  next[11] = index + (1 + zperiod);
  next[12] = index - (1 + zperiod);
  next[13] = index + (1 - zperiod);
  next[14] = index - (1 - zperiod);
  next[15] = index + (yperiod + zperiod);
  next[16] = index - (yperiod + zperiod);
  next[17] = index + (yperiod - zperiod);
  next[18] = index - (yperiod - zperiod);

  /* normalization factors enter in the back transformation */
  for (int i = 0; i < lbmodel.n_veloc; i++)
    m[i] = (1. / lbmodel.e[19][i]) * m[i];

  lbfluid[0][next[0]] = m[0] - m[4] + m[16];
  lbfluid[1][next[1]] =
      m[0] + m[1] + m[5] + m[6] - m[17] - m[18] - 2. * (m[10] + m[16]);
  lbfluid[2][next[2]] =
      m[0] - m[1] + m[5] + m[6] - m[17] - m[18] + 2. * (m[10] - m[16]);
  lbfluid[3][next[3]] =
      m[0] + m[2] - m[5] + m[6] + m[17] - m[18] - 2. * (m[11] + m[16]);
  lbfluid[4][next[4]] =
      m[0] - m[2] - m[5] + m[6] + m[17] - m[18] + 2. * (m[11] - m[16]);
  lbfluid[5][next[5]] = m[0] + m[3] - 2. * (m[6] + m[12] + m[16] - m[18]);
  lbfluid[6][next[6]] = m[0] - m[3] - 2. * (m[6] - m[12] + m[16] - m[18]);
  lbfluid[7][next[7]] = m[0] + m[1] + m[2] + m[4] + 2. * m[6] + m[7] + m[10] +
                        m[11] + m[13] + m[14] + m[16] + 2. * m[18];
  lbfluid[8][next[8]] = m[0] - m[1] - m[2] + m[4] + 2. * m[6] + m[7] - m[10] -
                        m[11] - m[13] - m[14] + m[16] + 2. * m[18];
  lbfluid[9][next[9]] = m[0] + m[1] - m[2] + m[4] + 2. * m[6] - m[7] + m[10] -
                        m[11] + m[13] - m[14] + m[16] + 2. * m[18];
  lbfluid[10][next[10]] = m[0] - m[1] + m[2] + m[4] + 2. * m[6] - m[7] - m[10] +
                          m[11] - m[13] + m[14] + m[16] + 2. * m[18];
  lbfluid[11][next[11]] = m[0] + m[1] + m[3] + m[4] + m[5] - m[6] + m[8] +
                          m[10] + m[12] - m[13] + m[15] + m[16] + m[17] - m[18];
  lbfluid[12][next[12]] = m[0] - m[1] - m[3] + m[4] + m[5] - m[6] + m[8] -
                          m[10] - m[12] + m[13] - m[15] + m[16] + m[17] - m[18];
  lbfluid[13][next[13]] = m[0] + m[1] - m[3] + m[4] + m[5] - m[6] - m[8] +
                          m[10] - m[12] - m[13] - m[15] + m[16] + m[17] - m[18];
  lbfluid[14][next[14]] = m[0] - m[1] + m[3] + m[4] + m[5] - m[6] - m[8] -
                          m[10] + m[12] + m[13] + m[15] + m[16] + m[17] - m[18];
  lbfluid[15][next[15]] = m[0] + m[2] + m[3] + m[4] - m[5] - m[6] + m[9] +
                          m[11] + m[12] - m[14] - m[15] + m[16] - m[17] - m[18];
  lbfluid[16][next[16]] = m[0] - m[2] - m[3] + m[4] - m[5] - m[6] + m[9] -
                          m[11] - m[12] + m[14] + m[15] + m[16] - m[17] - m[18];
  lbfluid[17][next[17]] = m[0] + m[2] - m[3] + m[4] - m[5] - m[6] - m[9] +
                          m[11] - m[12] - m[14] + m[15] + m[16] - m[17] - m[18];
  lbfluid[18][next[18]] = m[0] - m[2] + m[3] + m[4] - m[5] - m[6] - m[9] -
                          m[11] + m[12] + m[14] - m[15] + m[16] - m[17] - m[18];

  /* weights enter in the back transformation */
  for (int i = 0; i < lbmodel.n_veloc; i++)
    lbfluid[i][next[i]] *= lbmodel.w[i];
}
#endif // LB_ADAPTIVE

/* Collisions and streaming (push scheme) */
inline void lb_collide_stream() {
  Lattice::index_t index;
  double modes[19];

/* loop over all lattice cells (halo excluded) */
#ifdef LB_BOUNDARIES
  for (auto it = LBBoundaries::lbboundaries.begin();
       it != LBBoundaries::lbboundaries.end(); ++it) {
    (**it).reset_force();
  }
#endif // LB_BOUNDARIES

#ifdef LB_ADAPTIVE
  int level;
#ifdef LB_ADAPTIVE_GPU
  // first part of subcycling; coarse to fine
  const auto &forest_lb =
      p4est_utils_get_forest_info(forest_order::adaptive_LB);
  for (level = forest_lb.coarsest_level_global;
       level <= forest_lb.finest_level_global; ++level) {
    // populate halos on that level
    lbadapt_patches_populate_halos(level);

    // offload patches to GPU
    lbadapt_gpu_offload_data(level);

    // collide in complete patch, halo included
    lbadapt_gpu_execute_collision_kernel(level);

    // populate virtual quadrants
    // TODO: implement; nop in regular case
    lbadapt_gpu_execute_populate_virtuals_kernel(level);
  }
  ++n_lbsteps;
  // second part of subcycling; fine to coarse
  for (level = forest_lb.coarsest_level_global;
       level <= forest_lb.finest_level_global; ++level) {
    // update from virtual quadrants
    // TODO: implement; nop in regular case
    lbadapt_gpu_execute_update_from_virtuals_kernel(level);

    // stream in complete patch, halo included. Avoid leaving the patch
    lbadapt_gpu_execute_streaming_kernel(level);

#ifdef LB_BOUNDARIES
    // bounce back in complete patch, halo included. Avoid leaving the patch.
    lbadapt_gpu_execute_bounce_back_kernel(level);
#endif // LB_BOUNDARIES

    // retrieve patches
    lbadapt_gpu_retrieve_data(level);

    // ghost exchange
    std::vector<lbadapt_payload_t *> local_pointer(P8EST_QMAXLEVEL);
    std::vector<lbadapt_payload_t *> ghost_pointer(P8EST_QMAXLEVEL);
    prepare_ghost_exchange(lbadapt_local_data, local_pointer,
                           lbadapt_ghost_data, ghost_pointer);

    p4est_virtual_ghost_exchange_data_level(
        adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual,
        adapt_virtual_ghost, level, sizeof(lbadapt_payload_t),
        (void **)local_pointer.data(), (void **)ghost_pointer.data());
  }
#else // LB_ADAPTIVE_GPU
  int lvl_diff;

  // perform 1st half of subcycling here (process coarse before fine)
  const auto &forest_lb =
      p4est_utils_get_forest_info(forest_order::adaptive_LB);
  for (level = p4est_params.min_ref_level; level <= p4est_params.max_ref_level;
       ++level) {
    // level always relates to level of real cells
    lvl_diff = p4est_params.max_ref_level - level;
    if (n_lbsteps % (1 << lvl_diff) == 0) {
#ifdef COMM_HIDING
      lbadapt_collide(level, P8EST_TRAVERSE_LOCAL);
      p4est_utils_end_pending_communication(exc_status_lb, level);
      lbadapt_collide(level, P8EST_TRAVERSE_GHOST);
#else
      lbadapt_collide(level, P8EST_TRAVERSE_LOCALGHOST);
#endif
    }
  }
  // increment counter half way to keep coarse quadrants from streaming early
  ++n_lbsteps;

  // perform second half of subcycling here (process fine before coarse)
  for (level = p4est_params.max_ref_level; p4est_params.min_ref_level <= level;
       --level) {
    // level always relates to level of real cells
    lvl_diff = p4est_params.max_ref_level - level;
    if (n_lbsteps % (1 << lvl_diff) == 0) {
#ifdef COMM_HIDING
      lbadapt_update_populations_from_virtuals(level, P8EST_TRAVERSE_LOCAL);

      p4est_utils_end_pending_communication(exc_status_lb, level);
      p4est_utils_end_pending_communication(exc_status_lb, level + 1);
      lbadapt_update_populations_from_virtuals(level, P8EST_TRAVERSE_GHOST);
#else
      lbadapt_update_populations_from_virtuals(level,
                                               P8EST_TRAVERSE_LOCALGHOST);
#endif
#if 0
      lbadapt_stream(level);
      lbadapt_bounce_back(level);
#else
      lbadapt_stream_bounce_back(level);
#endif
      lbadapt_swap_pointers(level);

      // synchronize ghost data for next collision step

#ifdef COMM_HIDING
      p4est_utils_start_communication(exc_status_lb, level, lbadapt_local_data,
                                      lbadapt_ghost_data);
#else
      std::vector<lbadapt_payload_t *> local_pointer(P8EST_QMAXLEVEL);
      std::vector<lbadapt_payload_t *> ghost_pointer(P8EST_QMAXLEVEL);
      prepare_ghost_exchange(lbadapt_local_data, local_pointer,
                             lbadapt_ghost_data, ghost_pointer);
      p8est_virtual_ghost_exchange_data_level(
          adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual,
          adapt_virtual_ghost, level, sizeof(lbadapt_payload_t),
          (void **)local_pointer.data(), (void **)ghost_pointer.data());
#endif
    }
  }
#endif // LB_ADAPTIVE_GPU
#else  // LB_ADAPTIVE

#ifdef VIRTUAL_SITES_INERTIALESS_TRACERS
  // Safeguard the node forces so that we can later use them for the IBM
  // particle update
  // In the following loop the lbfields[XX].force are reset to zero
  // Safeguard the node forces so that we can later use them for the IBM
  // particle update In the following loop the lbfields[XX].force are reset to
  // zero
  for (int i = 0; i < lblattice.halo_grid_volume; ++i) {
    lbfields[i].force_density_buf[0] = lbfields[i].force_density[0];
    lbfields[i].force_density_buf[1] = lbfields[i].force_density[1];
    lbfields[i].force_density_buf[2] = lbfields[i].force_density[2];
  }
#endif

  index = lblattice.halo_offset;
  for (int z = 1; z <= lblattice.grid[2]; z++) {
    for (int y = 1; y <= lblattice.grid[1]; y++) {
      for (int x = 1; x <= lblattice.grid[0]; x++) {
// as we only want to apply this to non-boundary nodes we can throw out
// the if-clause if we have a non-bounded domain
#ifdef LB_BOUNDARIES
        if (!lbfields[index].boundary)
#endif // LB_BOUNDARIES
        {
          /* calculate modes locally */
          lb_calc_modes(index, modes);

          /* deterministic collisions */
          lb_relax_modes(index, modes);

          /* fluctuating hydrodynamics */
          if (lbpar.fluct)
            lb_thermalize_modes(index, modes);

          /* apply forces */
          lb_apply_forces(index, modes);

          lb_reset_force_densities(index);

          /* transform back to populations and streaming */
          lb_calc_n_from_modes_push(lbfluid_post, index, modes);
        }

        ++index; /* next node */
      }
      index += 2; /* skip halo region */
    }
    index += 2 * lblattice.halo_grid[0]; /* skip halo region */
  }

  /* exchange halo regions */
  halo_push_communication(lbfluid_post);

#ifdef LB_BOUNDARIES
  /* boundary conditions for links */
  LBBoundaries::lb_bounce_back(lbfluid_post);
#endif // LB_BOUNDARIES

  /* swap the pointers for old and new population fields */
  std::swap(lbfluid, lbfluid_post);

  /* halo region is invalid after update */
  lbpar.resend_halo = 1;
#endif // LB_ADAPTIVE
}

/***********************************************************************/
/** \name Update step for the lattice Boltzmann fluid                  */
/***********************************************************************/
/*@{*/
/*@}*/

/** Update the lattice Boltzmann fluid.
 *
 * This function is called from the integrator. Since the time step
 * for the lattice dynamics can be coarser than the MD time step, we
 * monitor the time since the last lattice update.
 */
void lattice_boltzmann_update() {
  int factor = (int)round(lbpar.tau / time_step);
#if defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
  if (n_part && 0 == n_lbsteps && adapt_p4est != nullptr) {
    p4est_utils_refine_around_particles();
  }
#endif

  fluidstep += 1;
  if (fluidstep >= factor) {
    fluidstep = 0;

    lb_collide_stream();
  }
}

/** Resets the forces on the fluid nodes */
void lb_reinit_force_densities() {
#ifndef LB_ADAPTIVE
  for (Lattice::index_t index = 0; index < lblattice.halo_grid_volume;
       index++) {
    lb_reset_force_densities(index);
  }
#ifdef LB_BOUNDARIES
  for (auto it = LBBoundaries::lbboundaries.begin();
       it != LBBoundaries::lbboundaries.end(); ++it) {
    (**it).reset_force();
  }
#endif // LB_BOUNDARIES
#endif
}

namespace {
template <typename Op>
void lattice_interpolation(Lattice const &lattice, Vector3d const &pos,
                           Op &&op) {
  Lattice::index_t node_index[8];
  double delta[6];

  /* determine elementary lattice cell surrounding the particle
     and the relative position of the particle in this cell */
  lattice.map_position_to_lattice(pos, node_index, delta);

  for (int z = 0; z < 2; z++) {
    for (int y = 0; y < 2; y++) {
      for (int x = 0; x < 2; x++) {
        auto &index = node_index[(z * 2 + y) * 2 + x];
        auto const w = delta[3 * x + 0] * delta[3 * y + 1] * delta[3 * z + 2];

        op(index, w);
      }
    }
  }
}
} // namespace

/***********************************************************************/
/** \name Coupling part */
/***********************************************************************/
/*@{*/

/** Coupling of a single particle to viscous fluid with Stokesian friction.
 *
 * Section II.C. Ahlrichs and Duenweg, JCP 111(17):8225 (1999)
 *
 * @param p          The coupled particle (Input).
 * @param force      Coupling force between particle and fluid (Output).
 */
inline void lb_viscous_coupling(Particle *p, double force[3],
                                bool ghost = false) {
#ifndef LB_ADAPTIVE_GPU
#ifdef LB_ADAPTIVE
  std::vector<lbadapt_payload_t *> payloads;
  std::vector<double> interpolation_weights;
  std::vector<int> quad_levels;
#endif // !LB_ADAPTIVE
  double *local_f, interpolated_u[3], delta_j[3];

#ifdef EXTERNAL_FORCES
  if (!(p->p.ext_flag & COORD_FIXED(0)) && !(p->p.ext_flag & COORD_FIXED(1)) &&
      !(p->p.ext_flag & COORD_FIXED(2))) {
    ONEPART_TRACE(if (p->p.identity == check_id) {
      fprintf(stderr, "%d: OPT: f = (%.3e,%.3e,%.3e)\n", this_node, p->f.f[0],
              p->f.f[1], p->f.f[2]);
    });
  }
#endif

  /* determine elementary lattice cell surrounding the particle
     and the relative position of the particle in this cell */
#ifdef LB_ADAPTIVE
  (ghost ? lbadapt_interpolate_pos_ghost : lbadapt_interpolate_pos_adapt)(
      p->r.p, payloads, interpolation_weights, quad_levels);
  if (payloads.empty()) {
    return;
  }

  P4EST_ASSERT(8 <= payloads.size() && payloads.size() <= 20);
  double h_max = p4est_params.h[p4est_params.max_ref_level];
#else
  double h_max = lbpar.agrid;
#endif // LB_ADAPTIVE

  /* calculate fluid velocity at particle's position
     this is done by linear interpolation
     (Eq. (11) Ahlrichs and Duenweg, JCP 111(17):8225 (1999)) */
#ifndef LB_ADAPTIVE
  lb_lbfluid_get_interpolated_velocity(p->r.p, interpolated_u);
#else
  lb_lbfluid_get_interpolated_velocity(p->r.p, interpolated_u, ghost);
#endif

  /* calculate viscous force
   * take care to rescale velocities with time_step and transform to MD units
   * (Eq. (9) Ahlrichs and Duenweg, JCP 111(17):8225 (1999)) */
  double velocity[3];
  velocity[0] = p->m.v[0];
  velocity[1] = p->m.v[1];
  velocity[2] = p->m.v[2];

  Vector3d v_drift = {interpolated_u[0], interpolated_u[1], interpolated_u[2]};
#ifdef ENGINE
  if (p->swim.swimming) {
    v_drift += p->swim.v_swim * p->r.calc_director();
    p->swim.v_center[0] = interpolated_u[0];
    p->swim.v_center[1] = interpolated_u[1];
    p->swim.v_center[2] = interpolated_u[2];
  }
#endif

#ifdef LB_ELECTROHYDRODYNAMICS
  v_drift += p->p.mu_E;
#endif

  force[0] = -lbpar.friction * (velocity[0] - v_drift[0]);
  force[1] = -lbpar.friction * (velocity[1] - v_drift[1]);
  force[2] = -lbpar.friction * (velocity[2] - v_drift[2]);

  force[0] = force[0] + p->lc.f_random[0];
  force[1] = force[1] + p->lc.f_random[1];
  force[2] = force[2] + p->lc.f_random[2];

  /* transform momentum transfer to lattice units
     (Eq. (12) Ahlrichs and Duenweg, JCP 111(17):8225 (1999)) */

  delta_j[0] = -force[0] * time_step * lbpar.tau / h_max;
  delta_j[1] = -force[1] * time_step * lbpar.tau / h_max;
  delta_j[2] = -force[2] * time_step * lbpar.tau / h_max;

#ifndef LB_ADAPTIVE
  lattice_interpolation(lblattice, p->r.p,
                        [&delta_j](Lattice::index_t index, double w) {
                          auto &node = lbfields[index];

                          node.force_density[0] += w * delta_j[0];
                          node.force_density[1] += w * delta_j[1];
                          node.force_density[2] += w * delta_j[2];
                        });
#else  // !LB_ADAPTIVE
  for (int x = 0; x < payloads.size(); ++x) {
    local_f = payloads[x]->lbfields.force_density;
    double level_fact = Utils::sqr(p4est_params.prefactors[quad_levels[x]]);

    local_f[0] += interpolation_weights[x] * delta_j[0] / level_fact;
    local_f[1] += interpolation_weights[x] * delta_j[1] / level_fact;
    local_f[2] += interpolation_weights[x] * delta_j[2] / level_fact;
  }
#endif // !LB_ADAPTIVE

  // map_position_to_lattice: position ... not inside a local plaquette in ...

#ifdef ENGINE
  if (p->swim.swimming) {
    // TODO: Fix LB mapping
    if (n_nodes > 1) {
      if (this_node == 0) {
        fprintf(stderr, "ERROR: Swimming is not compatible with Open MPI and "
                        "CPU LB on more than 1 node.\n");
        fprintf(stderr, "       Please use LB_GPU instead.\n");
      }
      errexit();
    }

    // calculate source position
    Vector3d source_position;
    double direction = double(p->swim.push_pull) * p->swim.dipole_length;
    source_position[0] = p->r.p[0] + direction * p->r.calc_director()[0];
    source_position[1] = p->r.p[1] + direction * p->r.calc_director()[1];
    source_position[2] = p->r.p[2] + direction * p->r.calc_director()[2];

    int corner[3] = {0, 0, 0};
    fold_position(source_position, corner);

    // get lattice cell corresponding to source position and interpolate
    // velocity
#ifdef LB_ADAPTIVE
    lbadapt_interpolate_pos_adapt(source_position, payloads,
                                  interpolation_weights, quad_levels);
#endif // !LB_ADAPTIVE
    lb_lbfluid_get_interpolated_velocity(Vector3d(source_position),
                                         p->swim.v_source.data());

    // calculate and set force at source position
    delta_j[0] = -p->swim.f_swim * p->r.calc_director()[0] * time_step *
                 lbpar.tau / lbpar.agrid;
    delta_j[1] = -p->swim.f_swim * p->r.calc_director()[1] * time_step *
                 lbpar.tau / lbpar.agrid;
    delta_j[2] = -p->swim.f_swim * p->r.calc_director()[2] * time_step *
                 lbpar.tau / lbpar.agrid;

#ifndef LB_ADAPTIVE
    lattice_interpolation(lblattice, source_position,
                          [&delta_j](Lattice::index_t index, double w) {
                            auto &node = lbfields[index];

                            node.force_density[0] += w * delta_j[0];
                            node.force_density[1] += w * delta_j[1];
                            node.force_density[2] += w * delta_j[2];
                          });
#else  // !LB_ADAPTIVE
    for (x = 0; x < payloads.size(); ++x) {
      local_f = payloads[x]->lbfields.force;
      double level_fact = Utils::sqr(prefactors[quad_levels[x]]);

      local_f[0] += interpolation_weights[x] * delta_j[0] / level_fact;
      local_f[1] += interpolation_weights[x] * delta_j[1] / level_fact;
      local_f[2] += interpolation_weights[x] * delta_j[2] / level_fact;
    }
#endif // !LB_ADAPTIVE
  }
#endif
#endif // !LB_ADAPTIVE_GPU
}

#ifndef LB_ADAPTIVE
namespace {
Vector3d node_u(Lattice::index_t index) {
#ifdef LB_BOUNDARIES
  if (lbfields[index].boundary) {
    return lbfields[index].slip_velocity;
  }
#endif // LB_BOUNDARIES

  double modes[19];
  lb_calc_modes(index, modes);
  auto const local_rho =
      lbpar.rho * lbpar.agrid * lbpar.agrid * lbpar.agrid + modes[0];

  return Vector3d{modes[1], modes[2], modes[3]} / local_rho;
}
} // namespace
#endif // !LB_ADAPTIVE

void lb_lbfluid_get_interpolated_velocity(const Vector3d &p, double *v) {
#ifndef LB_ADAPTIVE_GPU
#ifdef LB_ADAPTIVE
  // FIXME Port to GPU
  bool ghost = (this_node !=
                p4est_utils_pos_to_proc(forest_order::adaptive_LB, p.data()));
  lb_lbfluid_get_interpolated_velocity(p, v, ghost);
#endif // !LB_ADAPTIVE
#endif // !LB_ADAPTIVE_GPU
}

void lb_lbfluid_get_interpolated_velocity(const Vector3d &pos, double *v,
                                          bool ghost) {
#ifndef LB_ADAPTIVE_GPU
// FIXME Port to GPU
#ifndef LB_ADAPTIVE
  Vector3d interpolated_u{};

  /* calculate fluid velocity at particle's position
     this is done by linear interpolation
     (Eq. (11) Ahlrichs and Duenweg, JCP 111(17):8225 (1999)) */
  lattice_interpolation(lblattice, pos,
                        [&interpolated_u](Lattice::index_t index, double w) {
                          auto &node = lbfields[index];

                          interpolated_u += w * node_u(index);
                        });

  v[0] = interpolated_u[0];
  v[1] = interpolated_u[1];
  v[2] = interpolated_u[2];
  v[0] *= lbpar.agrid / lbpar.tau;
  v[1] *= lbpar.agrid / lbpar.tau;
  v[2] *= lbpar.agrid / lbpar.tau;
#else // !LB_ADAPTIVE
  lbadapt_payload_t *data;
  std::vector<lbadapt_payload_t *> payloads;
  std::vector<double> interpolation_weights;
  std::vector<int> quad_levels;

  double local_rho, local_j[3], interpolated_u[3];
  double modes[19];

  if (ghost) {
    lbadapt_interpolate_pos_ghost(pos, payloads, interpolation_weights,
                                  quad_levels);
  } else {
    lbadapt_interpolate_pos_adapt(pos, payloads, interpolation_weights,
                                  quad_levels);
  }
  double h_local = p4est_params.h[quad_levels[0]];
  double h_max = p4est_params.h[p4est_params.max_ref_level];

  for (int x = 0; x < payloads.size(); ++x) {
    data = payloads[x];
#ifdef LB_BOUNDARIES
    int bnd = data->lbfields.boundary;
    if (bnd) {
      local_rho = lbpar.rho * h_max * h_max * h_max;
      local_j[0] = lbpar.rho * h_max * h_max * h_max *
                   (*LBBoundaries::lbboundaries[bnd - 1]).velocity()[0]; // TODO
      local_j[1] = lbpar.rho * h_max * h_max * h_max *
                   (*LBBoundaries::lbboundaries[bnd - 1])
                       .velocity()[1]; // TODO This might not work properly
      local_j[2] = lbpar.rho * h_max * h_max * h_max *
                   (*LBBoundaries::lbboundaries[bnd - 1]).velocity()[2]; // TODO
    } else {
      lbadapt_calc_modes(data->lbfluid, modes);
      local_rho = lbpar.rho * h_max * h_max * h_max + modes[0];
      local_j[0] = modes[1];
      local_j[1] = modes[2];
      local_j[2] = modes[3];
    }
#else  // LB_BOUNDARIES
    lbadapt_calc_modes(data->lbfluid, modes);
    local_rho = lbpar.rho[0] * h_max * h_max * h_max + modes[0];
    local_j[0] = modes[1];
    local_j[1] = modes[2];
    local_j[2] = modes[3];
#endif // LB_BOUNDARIES
    interpolated_u[0] += interpolation_weights[x] * local_j[0] / (local_rho);
    interpolated_u[1] += interpolation_weights[x] * local_j[1] / (local_rho);
    interpolated_u[2] += interpolation_weights[x] * local_j[2] / (local_rho);
  }
#endif // LB_ADAPTIVE
#endif //! LB_ADAPTIVE_GPU
}

/** Calculate particle lattice interactions.
 * So far, only viscous coupling with Stokesian friction is
 * implemented.
 * Include all particle-lattice forces in this function.
 * The function is called from \ref force_calc.
 *
 * Parallelizing the fluid particle coupling is not straightforward
 * because drawing of random numbers makes the whole thing nonlocal.
 * One way to do it is to treat every particle only on one node, i.e.
 * the random numbers need not be communicated. The particles that are
 * not fully inside the local lattice are taken into account via their
 * ghost images on the neighbouring nodes. But this requires that the
 * correct values of the surrounding lattice nodes are available on
 * the respective node, which means that we have to communicate the
 * halo regions before treating the ghost particles. Moreover, after
 * determining the ghost couplings, we have to communicate back the
 * halo region such that all local lattice nodes have the correct values.
 * Thus two communication phases are involved which will most likely be
 * the bottleneck of the computation.
 *
 * Another way of dealing with the particle lattice coupling is to
 * treat a particle and all of it's images explicitly. This requires the
 * communication of the random numbers used in the calculation of the
 * coupling force. The problem is now that, if random numbers have to
 * be redrawn, we cannot efficiently determine which particles and which
 * images have to be re-calculated. We therefore go back to the outset
 * and go through the whole system again until no failure occurs during
 * such a sweep. In the worst case, this is very inefficient because
 * many things are recalculated although they actually don't need.
 * But we can assume that this happens extremely rarely and then we have
 * on average only one communication phase for the random numbers, which
 * probably makes this method preferable compared to the above one.
 */
void calc_particle_lattice_ia() {

  if (transfer_momentum) {
    double force[3];

#ifdef LB_ADAPTIVE
#ifdef DD_P4EST
#ifdef COMM_HIDING
    if (n_part)
      p4est_utils_end_pending_communication(exc_status_lb);
#endif // COMM_HIDING
#endif // DD_P4EST
#else  // LB_ADAPTIVE
    if (lbpar.resend_halo) { /* first MD step after last LB update */

      /* exchange halo regions (for fluid-particle coupling) */
      halo_communication(&update_halo_comm,
                         reinterpret_cast<char *>(lbfluid[0].data()));

#ifdef ADDITIONAL_CHECKS
      lb_check_halo_regions(lbfluid);
#endif

      /* halo is valid now */
      lbpar.resend_halo = 0;
    }
#endif // LB ADAPTIVE

    /* draw random numbers for local particles */
    for (auto &p : local_cells.particles()) {
      if (lb_coupl_pref2 > 0.0) {
#ifdef GAUSSRANDOM
        p.lc.f_random[0] = lb_coupl_pref2 * gaussian_random();
        p.lc.f_random[1] = lb_coupl_pref2 * gaussian_random();
        p.lc.f_random[2] = lb_coupl_pref2 * gaussian_random();
#elif defined(GAUSSRANDOMCUT)
        p.lc.f_random[0] = lb_coupl_pref2 * gaussian_random_cut();
        p.lc.f_random[1] = lb_coupl_pref2 * gaussian_random_cut();
        p.lc.f_random[2] = lb_coupl_pref2 * gaussian_random_cut();
#elif defined(FLATNOISE)
        p.lc.f_random[0] = lb_coupl_pref * (d_random() - 0.5);
        p.lc.f_random[1] = lb_coupl_pref * (d_random() - 0.5);
        p.lc.f_random[2] = lb_coupl_pref * (d_random() - 0.5);
#else // GAUSSRANDOM
#error No noise type defined for the CPU LB
#endif // GAUSSRANDOM
      } else {
        p.lc.f_random = {0.0, 0.0, 0.0};
      }

#ifdef ADDITIONAL_CHECKS
      rancounter += 3;
#endif // ADDITIONAL_CHECKS
    }

#ifdef ENGINE
    const int data_parts = GHOSTTRANS_COUPLING | GHOSTTRANS_SWIMMING;
#else
    const int data_parts = GHOSTTRANS_COUPLING;
#endif
#ifdef LB_ADAPTIVE
    // Clear coupling quads vector
    coupling_quads = std::vector<bool>(adapt_p4est->local_num_quadrants, false);
#endif

    /* communicate the random numbers */
    ghost_communicator(&cell_structure.exchange_ghosts_comm, data_parts);

    /* local cells */
    for (auto &p : local_cells.particles()) {
      if (!p.p.is_virtual || thermo_virtual) {
        lb_viscous_coupling(&p, force);

        /* add force to the particle */
        p.f.f[0] += force[0];
        p.f.f[1] += force[1];
        p.f.f[2] += force[2];

        ONEPART_TRACE(if (p.p.identity == check_id) {
          fprintf(stderr, "%d: OPT: LB f = (%.6e,%.3e,%.3e)\n", this_node,
                  p.f.f[0], p.f.f[1], p.f.f[2]);
        });
      }
    }

    /* ghost cells */
    for (auto &p : ghost_cells.particles()) {
#if !defined(DD_P4EST) && !defined(LB_ADAPTIVE)
      /* for ghost particles we have to check if they lie
       * in the range of the local lattice nodes */
      if (p.r.p[0] >= my_left[0] - 0.5 * lblattice.agrid[0] &&
          p.r.p[0] < my_right[0] + 0.5 * lblattice.agrid[0] &&
          p.r.p[1] >= my_left[1] - 0.5 * lblattice.agrid[1] &&
          p.r.p[1] < my_right[1] + 0.5 * lblattice.agrid[1] &&
          p.r.p[2] >= my_left[2] - 0.5 * lblattice.agrid[2] &&
          p.r.p[2] < my_right[2] + 0.5 * lblattice.agrid[2])
#endif
      {
        ONEPART_TRACE(if (p.p.identity == check_id) {
          fprintf(stderr, "%d: OPT: LB coupling of ghost particle:\n",
                  this_node);
        });
        if (!p.p.is_virtual || thermo_virtual) {
#ifndef DD_P4EST
          lb_viscous_coupling(&p, force);
#else
          lb_viscous_coupling(&p, force, true);
#endif
        }

        /* ghosts must not have the force added! */
        ONEPART_TRACE(if (p.p.identity == check_id) {
          fprintf(stderr, "%d: OPT: LB f = (%.6e,%.3e,%.3e)\n", this_node,
                  p.f.f[0], p.f.f[1], p.f.f[2]);
        });
      }
    }

#ifdef LB_ADAPTIVE
#ifdef DD_P4EST
    if (n_part) {
#ifdef COMM_HIDING
      p4est_utils_start_communication(exc_status_lb, -1, lbadapt_local_data,
                                      lbadapt_ghost_data);
#else
      std::vector<lbadapt_payload_t *> local_pointer(P8EST_QMAXLEVEL);
      std::vector<lbadapt_payload_t *> ghost_pointer(P8EST_QMAXLEVEL);
      prepare_ghost_exchange(lbadapt_local_data, local_pointer,
                             lbadapt_ghost_data, ghost_pointer);
      for (int level = p4est_params.min_ref_level;
           level <= p4est_params.max_ref_level; ++level) {
        p8est_virtual_ghost_exchange_data_level(
            adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual,
            adapt_virtual_ghost, level, sizeof(lbadapt_payload_t),
            (void **)local_pointer.data(), (void **)ghost_pointer.data());
      }
#endif
    }
#endif
#endif
  }
}

/***********************************************************************/

/** Calculate the average density of the fluid in the system.
 * This function has to be called after changing the density of
 * a local lattice site in order to set lbpar.rho consistently. */
void lb_calc_average_rho() {
  double rho, sum_rho;
#ifdef LB_ADAPTIVE
  rho = 0.0;
  p8est_iterate(adapt_p4est, nullptr, (void *)&rho, lbadapt_calc_local_rho,
                nullptr, nullptr, nullptr);
#else // LB_ADAPTIVE
  double local_rho;
  Lattice::index_t index;
  int x, y, z;

  rho = 0.0;
  local_rho = 0.0;
  index = 0;
  for (z = 1; z <= lblattice.grid[2]; z++) {
    for (y = 1; y <= lblattice.grid[1]; y++) {
      for (x = 1; x <= lblattice.grid[0]; x++) {
        lb_calc_local_rho(index, &rho);
        local_rho += rho;

        index++;
      }
      // skip halo region
      index += 2;
    }
    // skip halo region
    index += 2 * lblattice.halo_grid[0];
  }
#endif
  MPI_Allreduce(&rho, &sum_rho, 1, MPI_DOUBLE, MPI_SUM, comm_cart);

  /* calculate average density in MD units */
  // TODO!!!
  lbpar.rho = sum_rho / (box_l[0] * box_l[1] * box_l[2]);
}

/*@}*/

void print_fluid() {
#ifndef LB_ADAPTIVE
  for (int x = 0; x < lblattice.halo_grid[0]; ++x) {
    for (int y = 0; y < lblattice.halo_grid[1]; ++y) {
      for (int z = 0; z < lblattice.halo_grid[2]; ++z) {
        int index = get_linear_index(x, y, z, lblattice.halo_grid);
        for (int p = 0; p < lbmodel.n_veloc; ++p) {
          printf("x %d y %d z %d pop %d: %f\n", x, y, z, p, lbfluid[p][index]);
        }
      }
    }
  }
#endif
}

static int compare_buffers(double *buf1, double *buf2, int size) {
  int ret;
  if (memcmp(buf1, buf2, size) != 0) {
    runtimeErrorMsg() << "Halo buffers are not identical";
    ret = 1;
  } else {
    ret = 0;
  }
  return ret;
}

/** Checks consistency of the halo regions (ADDITIONAL_CHECKS)
    This function can be used as an additional check. It test whether the
    halo regions have been exchanged correctly.
*/
#ifndef LB_ADAPTIVE
void lb_check_halo_regions(const LB_Fluid &lbfluid) {
  Lattice::index_t index;
  int i, x, y, z, s_node, r_node, count = lbmodel.n_veloc;
  double *s_buffer, *r_buffer;
  MPI_Status status[2];

  r_buffer = (double *)Utils::malloc(count * sizeof(double));
  s_buffer = (double *)Utils::malloc(count * sizeof(double));

  if (PERIODIC(0)) {
    for (z = 0; z < lblattice.halo_grid[2]; ++z) {
      for (y = 0; y < lblattice.halo_grid[1]; ++y) {
        index = get_linear_index(0, y, z, lblattice.halo_grid);
        for (i = 0; i < lbmodel.n_veloc; i++)
          s_buffer[i] = lbfluid[i][index];

        s_node = node_neighbors[1];
        r_node = node_neighbors[0];
        if (n_nodes > 1) {
          MPI_Sendrecv(s_buffer, count, MPI_DOUBLE, r_node, REQ_HALO_CHECK,
                       r_buffer, count, MPI_DOUBLE, s_node, REQ_HALO_CHECK,
                       comm_cart, status);
          index =
              get_linear_index(lblattice.grid[0], y, z, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            s_buffer[i] = lbfluid[i][index];
          compare_buffers(s_buffer, r_buffer, count * sizeof(double));
        } else {
          index =
              get_linear_index(lblattice.grid[0], y, z, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            r_buffer[i] = lbfluid[i][index];
          if (compare_buffers(s_buffer, r_buffer, count * sizeof(double))) {
            std::cerr << "buffers differ in dir=" << 0 << " at index=" << index
                      << " y=" << y << " z=" << z << "\n";
          }
        }

        index =
            get_linear_index(lblattice.grid[0] + 1, y, z, lblattice.halo_grid);
        for (i = 0; i < lbmodel.n_veloc; i++)
          s_buffer[i] = lbfluid[i][index];

        s_node = node_neighbors[0];
        r_node = node_neighbors[1];
        if (n_nodes > 1) {
          MPI_Sendrecv(s_buffer, count, MPI_DOUBLE, r_node, REQ_HALO_CHECK,
                       r_buffer, count, MPI_DOUBLE, s_node, REQ_HALO_CHECK,
                       comm_cart, status);
          index = get_linear_index(1, y, z, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            s_buffer[i] = lbfluid[i][index];
          compare_buffers(s_buffer, r_buffer, count * sizeof(double));
        } else {
          index = get_linear_index(1, y, z, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            r_buffer[i] = lbfluid[i][index];
          if (compare_buffers(s_buffer, r_buffer, count * sizeof(double))) {
            std::cerr << "buffers differ in dir=0 at index=" << index
                      << " y=" << y << " z=" << z << "\n";
          }
        }
      }
    }
  }

  if (PERIODIC(1)) {
    for (z = 0; z < lblattice.halo_grid[2]; ++z) {
      for (x = 0; x < lblattice.halo_grid[0]; ++x) {
        index = get_linear_index(x, 0, z, lblattice.halo_grid);
        for (i = 0; i < lbmodel.n_veloc; i++)
          s_buffer[i] = lbfluid[i][index];

        s_node = node_neighbors[3];
        r_node = node_neighbors[2];
        if (n_nodes > 1) {
          MPI_Sendrecv(s_buffer, count, MPI_DOUBLE, r_node, REQ_HALO_CHECK,
                       r_buffer, count, MPI_DOUBLE, s_node, REQ_HALO_CHECK,
                       comm_cart, status);
          index =
              get_linear_index(x, lblattice.grid[1], z, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            s_buffer[i] = lbfluid[i][index];
          compare_buffers(s_buffer, r_buffer, count * sizeof(double));
        } else {
          index =
              get_linear_index(x, lblattice.grid[1], z, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            r_buffer[i] = lbfluid[i][index];
          if (compare_buffers(s_buffer, r_buffer, count * sizeof(double))) {
            std::cerr << "buffers differ in dir=1 at index=" << index
                      << " x=" << x << " z=" << z << "\n";
          }
        }
      }
      for (x = 0; x < lblattice.halo_grid[0]; ++x) {
        index =
            get_linear_index(x, lblattice.grid[1] + 1, z, lblattice.halo_grid);
        for (i = 0; i < lbmodel.n_veloc; i++)
          s_buffer[i] = lbfluid[i][index];

        s_node = node_neighbors[2];
        r_node = node_neighbors[3];
        if (n_nodes > 1) {
          MPI_Sendrecv(s_buffer, count, MPI_DOUBLE, r_node, REQ_HALO_CHECK,
                       r_buffer, count, MPI_DOUBLE, s_node, REQ_HALO_CHECK,
                       comm_cart, status);
          index = get_linear_index(x, 1, z, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            s_buffer[i] = lbfluid[i][index];
          compare_buffers(s_buffer, r_buffer, count * sizeof(double));
        } else {
          index = get_linear_index(x, 1, z, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            r_buffer[i] = lbfluid[i][index];
          if (compare_buffers(s_buffer, r_buffer, count * sizeof(double))) {
            std::cerr << "buffers differ in dir=1 at index=" << index
                      << " x=" << x << " z=" << z << "\n";
          }
        }
      }
    }
  }

  if (PERIODIC(2)) {
    for (y = 0; y < lblattice.halo_grid[1]; ++y) {
      for (x = 0; x < lblattice.halo_grid[0]; ++x) {
        index = get_linear_index(x, y, 0, lblattice.halo_grid);
        for (i = 0; i < lbmodel.n_veloc; i++)
          s_buffer[i] = lbfluid[i][index];

        s_node = node_neighbors[5];
        r_node = node_neighbors[4];
        if (n_nodes > 1) {
          MPI_Sendrecv(s_buffer, count, MPI_DOUBLE, r_node, REQ_HALO_CHECK,
                       r_buffer, count, MPI_DOUBLE, s_node, REQ_HALO_CHECK,
                       comm_cart, status);
          index =
              get_linear_index(x, y, lblattice.grid[2], lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            s_buffer[i] = lbfluid[i][index];
          compare_buffers(s_buffer, r_buffer, count * sizeof(double));
        } else {
          index =
              get_linear_index(x, y, lblattice.grid[2], lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            r_buffer[i] = lbfluid[i][index];
          if (compare_buffers(s_buffer, r_buffer, count * sizeof(double))) {
            std::cerr << "buffers differ in dir=2 at index=" << index
                      << " x=" << x << " y=" << y << " z=" << lblattice.grid[2]
                      << "\n";
          }
        }
      }
    }
    for (y = 0; y < lblattice.halo_grid[1]; ++y) {
      for (x = 0; x < lblattice.halo_grid[0]; ++x) {
        index =
            get_linear_index(x, y, lblattice.grid[2] + 1, lblattice.halo_grid);
        for (i = 0; i < lbmodel.n_veloc; i++)
          s_buffer[i] = lbfluid[i][index];

        s_node = node_neighbors[4];
        r_node = node_neighbors[5];
        if (n_nodes > 1) {
          MPI_Sendrecv(s_buffer, count, MPI_DOUBLE, r_node, REQ_HALO_CHECK,
                       r_buffer, count, MPI_DOUBLE, s_node, REQ_HALO_CHECK,
                       comm_cart, status);
          index = get_linear_index(x, y, 1, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            s_buffer[i] = lbfluid[i][index];
          compare_buffers(s_buffer, r_buffer, count * sizeof(double));
        } else {
          index = get_linear_index(x, y, 1, lblattice.halo_grid);
          for (i = 0; i < lbmodel.n_veloc; i++)
            r_buffer[i] = lbfluid[i][index];
          if (compare_buffers(s_buffer, r_buffer, count * sizeof(double))) {
            std::cerr << "buffers differ in dir=2 at index=" << index
                      << " x=" << x << " y=" << y << "\n";
          }
        }
      }
    }
  }

  free(r_buffer);
  free(s_buffer);
}
#endif

#endif // LB
