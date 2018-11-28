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
    Implementation of \ref initialize.hpp "initialize.hpp"
*/
#include "initialize.hpp"
#include "bonded_interactions/thermalized_bond.hpp"
#include "cells.hpp"
#include "communication.hpp"
#include "cuda_init.hpp"
#include "cuda_interface.hpp"
#include "dpd.hpp"
#include "electrostatics_magnetostatics/debye_hueckel.hpp"
#include "electrostatics_magnetostatics/elc.hpp"
#include "electrostatics_magnetostatics/icc.hpp" /* -iccp3m- */
#include "electrostatics_magnetostatics/maggs.hpp"
#include "electrostatics_magnetostatics/mmm1d.hpp"
#include "electrostatics_magnetostatics/mmm2d.hpp"
#include "electrostatics_magnetostatics/p3m-dipolar.hpp"
#include "electrostatics_magnetostatics/p3m.hpp"
#include "electrostatics_magnetostatics/p3m_gpu.hpp"
#include "electrostatics_magnetostatics/scafacos.hpp"
#include "energy.hpp"
#include "errorhandling.hpp"
#include "forces.hpp"
#include "ghmc.hpp"
#include "ghosts.hpp"
#include "global.hpp"
#include "grid.hpp"
#include "grid_based_algorithms/lb.hpp"
#include "grid_based_algorithms/lbboundaries.hpp"
#include "grid_based_algorithms/lbgpu.hpp"
#include "lattice.hpp"
#include "metadynamics.hpp"
#include "nemd.hpp"
#include "nonbonded_interactions/reaction_field.hpp"
#include "npt.hpp"
#include "nsquare.hpp"
#include "observables/Observable.hpp"
#include "p4est_utils.hpp"
#include "partCfg_global.hpp"
#include "particle_data.hpp"
#include "pressure.hpp"
#include "random.hpp"
#include "rattle.hpp"
#include "reaction_ensemble.hpp"
#include "rotation.hpp"
#include "statistics.hpp"
#include "swimmer_reaction.hpp"
#include "thermostat.hpp"
#include "utils.hpp"
#include "virtual_sites.hpp"

#include "utils/mpi/all_compare.hpp"

#ifdef DD_P4EST
#include <p4est_to_p8est.h>
#endif

/** whether the thermostat has to be reinitialized before integration */
static int reinit_thermo = 1;
static int reinit_electrostatics = 0;
static int reinit_magnetostatics = 0;
#ifdef LB_GPU
static int lb_reinit_particles_gpu = 1;
#endif

#ifdef CUDA
static int reinit_particle_comm_gpu = 1;
#endif

void on_program_start() {
  EVENT_TRACE(fprintf(stderr, "%d: on_program_start\n", this_node));

#ifdef CUDA
  cuda_init();
#endif

  /*
    call the initialization of the modules here
  */
  Random::init_random();

  init_node_grid();

  // auto verbosity = SC_LP_ESSENTIAL;
  auto verbosity = SC_LP_PRODUCTION;
  // auto verbosity = SC_LP_VERBOSE;
  sc_init (comm_cart, 1, 1, nullptr, verbosity);
#if (defined(DD_P4EST) || defined(LB_ADAPTIVE))
  p4est_init(nullptr, verbosity);
#ifdef DD_P4EST
  topology_init(CELL_STRUCTURE_P4EST, &local_cells);
#endif
#else
  /* initially go for domain decomposition */
  topology_init(CELL_STRUCTURE_DOMDEC, &local_cells);
#endif

  // initialize stats
  sc_stats_init(&stats[LB_STEP_00], "Coarse LB step 00");
  sc_stats_init(&stats[LB_STEP_01], "Coarse LB step 01");
  sc_stats_init(&stats[LB_STEP_02], "Coarse LB step 02");
  sc_stats_init(&stats[LB_STEP_03], "Coarse LB step 03");
  sc_stats_init(&stats[LB_STEP_04], "Coarse LB step 04");
  sc_stats_init(&stats[LB_STEP_05], "Coarse LB step 05");
  sc_stats_init(&stats[LB_STEP_06], "Coarse LB step 06");
  sc_stats_init(&stats[LB_STEP_07], "Coarse LB step 07");
  sc_stats_init(&stats[LB_STEP_08], "Coarse LB step 08");
  sc_stats_init(&stats[LB_STEP_09], "Coarse LB step 09");
  sc_stats_init(&stats[MD_STEP_00], "Coarse MD step 00");
  sc_stats_init(&stats[MD_STEP_01], "Coarse MD step 01");
  sc_stats_init(&stats[MD_STEP_02], "Coarse MD step 02");
  sc_stats_init(&stats[MD_STEP_03], "Coarse MD step 03");
  sc_stats_init(&stats[MD_STEP_04], "Coarse MD step 04");
  sc_stats_init(&stats[MD_STEP_05], "Coarse MD step 05");
  sc_stats_init(&stats[MD_STEP_06], "Coarse MD step 06");
  sc_stats_init(&stats[MD_STEP_07], "Coarse MD step 07");
  sc_stats_init(&stats[MD_STEP_08], "Coarse MD step 08");
  sc_stats_init(&stats[MD_STEP_09], "Coarse MD step 09");
#ifdef LB_ADAPTIVE
  sc_stats_init(&stats[GRID_CHANGE_00], "Grid change step 00");
  sc_stats_init(&stats[GRID_CHANGE_01], "Grid change step 01");
  sc_stats_init(&stats[GRID_CHANGE_02], "Grid change step 02");
  sc_stats_init(&stats[GRID_CHANGE_03], "Grid change step 03");
  sc_stats_init(&stats[GRID_CHANGE_04], "Grid change step 04");
  sc_stats_init(&stats[GRID_CHANGE_05], "Grid change step 05");
  sc_stats_init(&stats[GRID_CHANGE_06], "Grid change step 06");
  sc_stats_init(&stats[GRID_CHANGE_07], "Grid change step 07");
  sc_stats_init(&stats[GRID_CHANGE_08], "Grid change step 08");
  sc_stats_init(&stats[GRID_CHANGE_09], "Grid change step 09");
#endif
  sc_stats_init(&stats[FLUPS_LB_00], "Fluid lattice updates per second 00");
  sc_stats_init(&stats[FLUPS_LB_01], "Fluid lattice updates per second 01");
  sc_stats_init(&stats[FLUPS_LB_02], "Fluid lattice updates per second 02");
  sc_stats_init(&stats[FLUPS_LB_03], "Fluid lattice updates per second 03");
  sc_stats_init(&stats[FLUPS_LB_04], "Fluid lattice updates per second 04");
  sc_stats_init(&stats[FLUPS_LB_05], "Fluid lattice updates per second 05");
  sc_stats_init(&stats[FLUPS_LB_06], "Fluid lattice updates per second 06");
  sc_stats_init(&stats[FLUPS_LB_07], "Fluid lattice updates per second 07");
  sc_stats_init(&stats[FLUPS_LB_08], "Fluid lattice updates per second 08");
  sc_stats_init(&stats[FLUPS_LB_09], "Fluid lattice updates per second 09");
  sc_stats_init(&stats[PUPS_MD_00], "Particle updates per second 00");
  sc_stats_init(&stats[PUPS_MD_01], "Particle updates per second 01");
  sc_stats_init(&stats[PUPS_MD_02], "Particle updates per second 02");
  sc_stats_init(&stats[PUPS_MD_03], "Particle updates per second 03");
  sc_stats_init(&stats[PUPS_MD_04], "Particle updates per second 04");
  sc_stats_init(&stats[PUPS_MD_05], "Particle updates per second 05");
  sc_stats_init(&stats[PUPS_MD_06], "Particle updates per second 06");
  sc_stats_init(&stats[PUPS_MD_07], "Particle updates per second 07");
  sc_stats_init(&stats[PUPS_MD_08], "Particle updates per second 08");
  sc_stats_init(&stats[PUPS_MD_09], "Particle updates per second 09");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_00], "Number of local cells step 00");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_01], "Number of local cells step 01");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_02], "Number of local cells step 02");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_03], "Number of local cells step 03");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_04], "Number of local cells step 04");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_05], "Number of local cells step 05");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_06], "Number of local cells step 06");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_07], "Number of local cells step 07");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_08], "Number of local cells step 08");
  sc_stats_init(&stats[NCELLS_LB_LOCAL_09], "Number of local cells step 09");
  sc_stats_init(&stats[NCELLS_LB_GHOST_00], "Number of ghost cells step 00");
  sc_stats_init(&stats[NCELLS_LB_GHOST_01], "Number of ghost cells step 01");
  sc_stats_init(&stats[NCELLS_LB_GHOST_02], "Number of ghost cells step 02");
  sc_stats_init(&stats[NCELLS_LB_GHOST_03], "Number of ghost cells step 03");
  sc_stats_init(&stats[NCELLS_LB_GHOST_04], "Number of ghost cells step 04");
  sc_stats_init(&stats[NCELLS_LB_GHOST_05], "Number of ghost cells step 05");
  sc_stats_init(&stats[NCELLS_LB_GHOST_06], "Number of ghost cells step 06");
  sc_stats_init(&stats[NCELLS_LB_GHOST_07], "Number of ghost cells step 07");
  sc_stats_init(&stats[NCELLS_LB_GHOST_08], "Number of ghost cells step 08");
  sc_stats_init(&stats[NCELLS_LB_GHOST_09], "Number of ghost cells step 09");
  sc_stats_init(&stats[NPART_LOCAL_00], "Number of local particles step 00");
  sc_stats_init(&stats[NPART_LOCAL_01], "Number of local particles step 01");
  sc_stats_init(&stats[NPART_LOCAL_02], "Number of local particles step 02");
  sc_stats_init(&stats[NPART_LOCAL_03], "Number of local particles step 03");
  sc_stats_init(&stats[NPART_LOCAL_04], "Number of local particles step 04");
  sc_stats_init(&stats[NPART_LOCAL_05], "Number of local particles step 05");
  sc_stats_init(&stats[NPART_LOCAL_06], "Number of local particles step 06");
  sc_stats_init(&stats[NPART_LOCAL_07], "Number of local particles step 07");
  sc_stats_init(&stats[NPART_LOCAL_08], "Number of local particles step 08");
  sc_stats_init(&stats[NPART_LOCAL_09], "Number of local particles step 09");
  sc_stats_init(&stats[NPART_GHOST_00], "Number of ghost particles step 00");
  sc_stats_init(&stats[NPART_GHOST_01], "Number of ghost particles step 01");
  sc_stats_init(&stats[NPART_GHOST_02], "Number of ghost particles step 02");
  sc_stats_init(&stats[NPART_GHOST_03], "Number of ghost particles step 03");
  sc_stats_init(&stats[NPART_GHOST_04], "Number of ghost particles step 04");
  sc_stats_init(&stats[NPART_GHOST_05], "Number of ghost particles step 05");
  sc_stats_init(&stats[NPART_GHOST_06], "Number of ghost particles step 06");
  sc_stats_init(&stats[NPART_GHOST_07], "Number of ghost particles step 07");
  sc_stats_init(&stats[NPART_GHOST_08], "Number of ghost particles step 08");
  sc_stats_init(&stats[NPART_GHOST_09], "Number of ghost particles step 09");

  n_integrate_calls = 0;

#ifdef P3M
  p3m_pre_init();
#endif
#ifdef DP3M
  dp3m_pre_init();
#endif

#ifdef LB_GPU
  if (this_node == 0) {
    //   lb_pre_init_gpu();
  }
#endif

#ifdef SWIMMER_REACTIONS
  reaction.eq_rate = 0.0;
  reaction.sing_mult = 0;
  reaction.swap = 0;
#endif

  /* call all initializations to do only on the master node here.  */
  if (this_node == 0) {
    /* interaction_data.c: make sure 0<->0 ia always exists */
    make_particle_type_exist(0);
  }
}

void on_integration_start() {
  EVENT_TRACE(fprintf(stderr, "%d: on_integration_start\n", this_node));
  INTEG_TRACE(fprintf(
      stderr,
      "%d: on_integration_start: reinit_thermo = %d, resort_particles=%d\n",
      this_node, reinit_thermo, get_resort_particles()));

  /********************************************/
  /* sanity checks                            */
  /********************************************/

  integrator_sanity_checks();
#ifdef NPT
  integrator_npt_sanity_checks();
#endif
  interactions_sanity_checks();
#ifdef SWIMMER_REACTIONS
  reactions_sanity_checks();
#endif
#ifdef LB
  if (lattice_switch & LATTICE_LB) {
    lb_sanity_checks();
  }
#endif
#ifdef LB_GPU
  if (lattice_switch & LATTICE_LB_GPU) {
    lb_GPU_sanity_checks();
  }
#endif

  /********************************************/
  /* end sanity checks                        */
  /********************************************/

#ifdef LB_GPU
  if (lattice_switch & LATTICE_LB_GPU && this_node == 0) {
    if (lb_reinit_particles_gpu) {
      lb_realloc_particles_gpu();
      lb_reinit_particles_gpu = 0;
    }
  }
#endif

#ifdef CUDA
  if (reinit_particle_comm_gpu) {
    gpu_change_number_of_part_to_comm();
    reinit_particle_comm_gpu = 0;
  }
  MPI_Bcast(gpu_get_global_particle_vars_pointer_host(),
            sizeof(CUDA_global_part_vars), MPI_BYTE, 0, comm_cart);
#endif

#ifdef METADYNAMICS
  meta_init();
#endif

  /* Prepare the thermostat */
  if (reinit_thermo) {
    thermo_init();
    reinit_thermo = 0;
    recalc_forces = 1;
  }

  /* Ensemble preparation: NVT or NPT */
  integrate_ensemble_init();

  /* Update particle and observable information for routines in statistics.cpp
   */
  invalidate_obs();
  partCfg().invalidate();
  invalidate_fetch_cache();

#ifdef ADDITIONAL_CHECKS

  if (!Utils::Mpi::all_compare(comm_cart, cell_structure.type)) {
    runtimeErrorMsg() << "Nodes disagree about cell system type.";
  }

  if (!Utils::Mpi::all_compare(comm_cart, get_resort_particles())) {
    runtimeErrorMsg() << "Nodes disagree about resort type.";
  }

  if (!Utils::Mpi::all_compare(comm_cart, cell_structure.use_verlet_list)) {
    runtimeErrorMsg() << "Nodes disagree about use of verlet lists.";
  }

#ifdef ELECTROSTATICS
  if (!Utils::Mpi::all_compare(comm_cart, coulomb.method))
    runtimeErrorMsg() << "Nodes disagree about Coulomb long range method";
#endif
#ifdef DIPOLES
  if (!Utils::Mpi::all_compare(comm_cart, coulomb.Dmethod))
    runtimeErrorMsg() << "Nodes disagree about dipolar long range method";
#endif
  check_global_consistency();
#endif /* ADDITIONAL_CHECKS */

  on_observable_calc();
}

void on_observable_calc() {
  EVENT_TRACE(fprintf(stderr, "%d: on_observable_calc\n", this_node));
  /* Prepare particle structure: Communication step: number of ghosts and ghost
   * information */

  cells_update_ghosts();

#ifdef ELECTROSTATICS
  if (reinit_electrostatics) {
    EVENT_TRACE(fprintf(stderr, "%d: reinit_electrostatics\n", this_node));
    switch (coulomb.method) {
#ifdef P3M
    case COULOMB_ELC_P3M:
    case COULOMB_P3M_GPU:
    case COULOMB_P3M:
      EVENT_TRACE(
          fprintf(stderr, "%d: p3m_count_charged_particles\n", this_node));
      p3m_count_charged_particles();
      break;
#endif
    case COULOMB_MAGGS:
      maggs_init();
      break;
    default:
      break;
    }
    reinit_electrostatics = 0;
  }
#endif /*ifdef ELECTROSTATICS */

#ifdef DIPOLES
  if (reinit_magnetostatics) {
    EVENT_TRACE(fprintf(stderr, "%d: reinit_magnetostatics\n", this_node));
    switch (coulomb.Dmethod) {
#ifdef DP3M
    case DIPOLAR_MDLC_P3M:
    // fall through
    case DIPOLAR_P3M:
      dp3m_count_magnetic_particles();
      break;
#endif
    default:
      break;
    }
    reinit_magnetostatics = 0;
  }
#endif /*ifdef ELECTROSTATICS */
}

void on_particle_charge_change() {
  reinit_electrostatics = 1;
  invalidate_obs();

  /* the particle information is no longer valid */
  partCfg().invalidate();
}

void on_particle_change() {
  EVENT_TRACE(fprintf(stderr, "%d: on_particle_change\n", this_node));

  set_resort_particles(Cells::RESORT_LOCAL);
  reinit_electrostatics = 1;
  reinit_magnetostatics = 1;

#ifdef LB_GPU
  lb_reinit_particles_gpu = 1;
#endif
#ifdef CUDA
  reinit_particle_comm_gpu = 1;
#endif
  invalidate_obs();

  /* the particle information is no longer valid */
  partCfg().invalidate();

  /* the particle information is no longer valid */
  invalidate_fetch_cache();
}

void on_coulomb_change() {
  EVENT_TRACE(fprintf(stderr, "%d: on_coulomb_change\n", this_node));
  invalidate_obs();

#ifdef ELECTROSTATICS
  switch (coulomb.method) {
  case COULOMB_DH:
    break;
#ifdef P3M
#ifdef CUDA
  case COULOMB_P3M_GPU:
    p3m_gpu_init(p3m.params.cao, p3m.params.mesh, p3m.params.alpha);
    break;
#endif
  case COULOMB_ELC_P3M:
    ELC_init();
  // fall through
  case COULOMB_P3M:
    p3m_init();
    break;
#endif
  case COULOMB_MMM1D:
    MMM1D_init();
    break;
  case COULOMB_MMM2D:
    MMM2D_init();
    break;
  case COULOMB_MAGGS:
    maggs_init();
    /* Maggs electrostatics needs ghost velocities */
    on_ghost_flags_change();
    break;
  default:
    break;
  }
#endif /* ELECTROSTATICS */

#ifdef DIPOLES
  switch (coulomb.Dmethod) {
#ifdef DP3M
  case DIPOLAR_MDLC_P3M:
  // fall through
  case DIPOLAR_P3M:
    dp3m_init();
    break;
#endif
  default:
    break;
  }
#endif /* ifdef DIPOLES */

  /* all Coulomb methods have a short range part, aka near field
     correction. Even in case of switching off, we should call this,
     since the required cutoff might have reduced. */
  on_short_range_ia_change();

#ifdef CUDA
  reinit_particle_comm_gpu = 1;
#endif
  recalc_forces = 1;
}

void on_short_range_ia_change() {
  EVENT_TRACE(fprintf(stderr, "%d: on_short_range_ia_changes\n", this_node));
  invalidate_obs();

  recalc_maximal_cutoff();
  cells_on_geometry_change(0);

  recalc_forces = 1;
}

void on_constraint_change() {
  EVENT_TRACE(fprintf(stderr, "%d: on_constraint_change\n", this_node));
  invalidate_obs();
  recalc_forces = 1;
}

void on_lbboundary_change() {
  EVENT_TRACE(fprintf(stderr, "%d: on_lbboundary_change\n", this_node));
  invalidate_obs();

#ifdef LB_BOUNDARIES
  if (lattice_switch & LATTICE_LB) {
    LBBoundaries::lb_init_boundaries();
  }
#endif

#ifdef LB_BOUNDARIES_GPU
  if (this_node == 0) {
    if (lattice_switch & LATTICE_LB_GPU) {
      LBBoundaries::lb_init_boundaries();
    }
  }
#endif

  recalc_forces = 1;
}

void on_resort_particles() {
  EVENT_TRACE(fprintf(stderr, "%d: on_resort_particles\n", this_node));
#ifdef ELECTROSTATICS
  switch (coulomb.method) {
#ifdef P3M
  case COULOMB_ELC_P3M:
    ELC_on_resort_particles();
    break;
#endif
  case COULOMB_MMM2D:
    MMM2D_on_resort_particles();
    break;
  default:
    break;
  }
#endif /* ifdef ELECTROSTATICS */

  /* DIPOLAR interactions so far don't need this */

  recalc_forces = 1;
}

void on_boxl_change() {
  EVENT_TRACE(fprintf(stderr, "%d: on_boxl_change\n", this_node));

  grid_changed_box_l();
  /* Electrostatics cutoffs mostly depend on the system size,
     therefore recalculate them. */
  recalc_maximal_cutoff();
  cells_on_geometry_change(0);

/* Now give methods a chance to react to the change in box length */
#ifdef ELECTROSTATICS
  switch (coulomb.method) {
#ifdef P3M
  case COULOMB_ELC_P3M:
    ELC_init();
  // fall through
  case COULOMB_P3M_GPU:
  case COULOMB_P3M:
    p3m_scaleby_box_l();
    break;
#endif
  case COULOMB_MMM1D:
    MMM1D_init();
    break;
  case COULOMB_MMM2D:
    MMM2D_init();
    break;
  case COULOMB_MAGGS:
    maggs_init();
    break;
#ifdef SCAFACOS
  case COULOMB_SCAFACOS:
    Scafacos::update_system_params();
    break;
#endif
  default:
    break;
  }
#endif

#ifdef DIPOLES
  switch (coulomb.Dmethod) {
#ifdef DP3M
  case DIPOLAR_MDLC_P3M:
  // fall through
  case DIPOLAR_P3M:
    dp3m_scaleby_box_l();
    break;
#endif
#ifdef SCAFACOS
  case DIPOLAR_SCAFACOS:
    Scafacos::update_system_params();
    break;
#endif
  default:
    break;
  }
#endif

#ifdef LB
  if (lattice_switch & LATTICE_LB) {
    lb_init();
#ifdef LB_BOUNDARIES
    LBBoundaries::lb_init_boundaries();
#endif
  }
#endif
}

void on_cell_structure_change(bool omitLBinit) {
  EVENT_TRACE(fprintf(stderr, "%d: on_cell_structure_change\n", this_node));

/* Now give methods a chance to react to the change in cell
   structure.  Most ES methods need to reinitialize, as they depend
   on skin, node grid and so on. Only for a change in box length we
   have separate, faster methods, as this might happen frequently
   in a NpT simulation. */
#ifdef ELECTROSTATICS
  switch (coulomb.method) {
  case COULOMB_DH:
    break;
#ifdef P3M
  case COULOMB_ELC_P3M:
    ELC_init();
  // fall through
  case COULOMB_P3M:
    p3m_init();
    break;
  case COULOMB_P3M_GPU:
    break;
#endif
  case COULOMB_MMM1D:
    MMM1D_init();
    break;
  case COULOMB_MMM2D:
    MMM2D_init();
    break;
  case COULOMB_MAGGS:
    maggs_init();
    /* Maggs electrostatics needs ghost velocities */
    on_ghost_flags_change();
    break;
  default:
    break;
  }
#endif /* ifdef ELECTROSTATICS */

#ifdef DIPOLES
  switch (coulomb.Dmethod) {
#ifdef DP3M
  case DIPOLAR_MDLC_P3M:
  // fall through
  case DIPOLAR_P3M:
    dp3m_init();
    break;
#endif
  default:
    break;
  }
#endif /* ifdef DIPOLES */

#ifdef LB
  if (!omitLBinit && (lattice_switch & LATTICE_LB)) {
    lb_init();
  }
#endif
}

void on_temperature_change() {
  EVENT_TRACE(fprintf(stderr, "%d: on_temperature_change\n", this_node));

#ifdef LB
  if (lattice_switch & LATTICE_LB) {
    lb_reinit_parameters();
  }
#endif
#ifdef LB_GPU
  if (this_node == 0) {
    if (lattice_switch & LATTICE_LB_GPU) {
      lb_reinit_parameters_gpu();
    }
  }
#endif
}

void on_parameter_change(int field) {
  EVENT_TRACE(
      fprintf(stderr, "%d: shon_parameter_change %d\n", this_node, field));

  switch (field) {
  case FIELD_BOXL:
    on_boxl_change();
    break;
  case FIELD_MIN_GLOBAL_CUT:
    recalc_maximal_cutoff();
    cells_on_geometry_change(0);
    break;
  case FIELD_SKIN:
    cells_on_geometry_change(0);
    break;
  case FIELD_PERIODIC:
#ifdef SCAFACOS
#ifdef ELECTROSTATICS
    if (coulomb.method == COULOMB_SCAFACOS) {
      Scafacos::update_system_params();
    }
#endif
#ifdef DIPOLES
    if (coulomb.Dmethod == DIPOLAR_SCAFACOS) {
      Scafacos::update_system_params();
    }
#endif

#endif
    cells_on_geometry_change(CELL_FLAG_GRIDCHANGED);
    break;
  case FIELD_NODEGRID:
    grid_changed_n_nodes();
    cells_on_geometry_change(CELL_FLAG_GRIDCHANGED);
    break;
  case FIELD_MINNUMCELLS:
  case FIELD_MAXNUMCELLS:
    cells_re_init(CELL_STRUCTURE_CURRENT);
    break;
  case FIELD_TEMPERATURE:
    on_temperature_change();
    reinit_thermo = 1;
    break;
  case FIELD_TIMESTEP:
#ifdef LB_GPU
    if (this_node == 0) {
      if (lattice_switch & LATTICE_LB_GPU) {
        lb_reinit_parameters_gpu();
      }
    }
#endif
#ifdef LB
    if (lattice_switch & LATTICE_LB) {
      lb_reinit_parameters();
    }
#endif
  case FIELD_LANGEVIN_GAMMA:
  case FIELD_LANGEVIN_GAMMA_ROTATION:
  case FIELD_NPTISO_G0:
  case FIELD_NPTISO_GV:
  case FIELD_NPTISO_PISTON:
    reinit_thermo = 1;
    break;
#ifdef NPT
  case FIELD_INTEG_SWITCH:
    if (integ_switch != INTEG_METHOD_NPT_ISO)
      nptiso.invalidate_p_vel = 1;
    break;
#endif
  case FIELD_THERMO_SWITCH:
    /* DPD needs ghost velocities, other thermostats not */
    on_ghost_flags_change();
#ifdef DPD
    if (not(thermo_switch & THERMO_DPD)) {
      dpd_switch_off();
    }
#endif
    break;
#ifdef LB
  case FIELD_LATTICE_SWITCH:
    /* LB needs ghost velocities */
    on_ghost_flags_change();
    break;
#endif
  case FIELD_FORCE_CAP:
    /* If the force cap changed, forces are invalid */
    invalidate_obs();
    recalc_forces = 1;
    break;
  case FIELD_RIGIDBONDS:
    /* Rattle bonds needs ghost velocities */
    on_ghost_flags_change();
    break;
  case FIELD_THERMALIZEDBONDS:
    /* Thermalized distance bonds needs ghost velocities */
    on_ghost_flags_change();
    break;
  }
}

#ifdef LB
void on_lb_params_change(int field) {
  EVENT_TRACE(fprintf(stderr, "%d: on_lb_params_change\n", this_node));

  if (field == LBPAR_AGRID) {
    lb_init();
  }
  if (field == LBPAR_DENSITY) {
    lb_reinit_fluid();
  }
  lb_reinit_parameters();
}
#endif

#if defined(LB) || defined(LB_GPU)
void on_lb_params_change_gpu(int field) {
  EVENT_TRACE(fprintf(stderr, "%d: on_lb_params_change_gpu\n", this_node));

#ifdef LB_GPU
  if (field == LBPAR_AGRID) {
    lb_init_gpu();
#ifdef LB_BOUNDARIES_GPU
    LBBoundaries::lb_init_boundaries();
#endif
  }
  if (field == LBPAR_DENSITY) {
    lb_reinit_fluid_gpu();
  }

  lb_reinit_parameters_gpu();
#endif
}
#endif

void on_ghost_flags_change() {
  EVENT_TRACE(fprintf(stderr, "%d: on_ghost_flags_change\n", this_node));
  /* that's all we change here */
  extern int ghosts_have_v;

  ghosts_have_v = 0;

/* DPD and LB need also ghost velocities */
#ifdef LB
  if (lattice_switch & LATTICE_LB)
    ghosts_have_v = 1;
#endif
#ifdef BOND_CONSTRAINT
  if (n_rigidbonds)
    ghosts_have_v = 1;
#endif
#ifdef ELECTROSTATICS
  /* Maggs electrostatics needs ghost velocities too */
  if (coulomb.method == COULOMB_MAGGS)
    ghosts_have_v = 1;
#endif
#ifdef DPD
  // maybe we have to add a new global to differ between compile in and actual
  // use.
  if (thermo_switch & THERMO_DPD)
    ghosts_have_v = 1;
#endif
#ifdef VIRTUAL_SITES
  // If they have velocities, VIRUTAL_SITES need v to update v of virtual sites
  if (virtual_sites()->get_have_velocity()) {
    ghosts_have_v = 1;
  };
#endif
  // THERMALIZED_DIST_BOND needs v to calculate v_com and v_dist for thermostats
  if (n_thermalized_bonds)
    ghosts_have_v = 1;
}
