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
#include "lb-adaptive_tcl.hpp"
#include "communication.hpp"
#include "lb-boundaries.hpp"

#include <stdio.h>

#ifdef LB_ADAPTIVE
int tclcommand_setup_grid(ClientData data, Tcl_Interp *interp, int argc,
                          char **argv) {
  /* container for parameters */
  int level = 0;

  /* verify input */
  if (argc != 2) {
    Tcl_AppendResult(interp, "Setting uniform refinement requires one "
                             "parameter, specifying refinement level.",
                     (char *)NULL);
    return TCL_ERROR;
  }

  if (!ARG_IS_I(1, level)) {
    Tcl_AppendResult(
        interp, "uniform refinement needs 1 parameter of type and meaning:\n",
        (char *)NULL);
    Tcl_AppendResult(interp, "INT\n", (char *)NULL);
    Tcl_AppendResult(interp, "<refinement_level>\n", (char *)NULL);
    return TCL_ERROR;
  }

  /* check input for semantic correctness */
  if ((level > 18) || (level < 0)) {
    Tcl_AppendResult(interp, "allowed refinement levels are [0, 18]\n",
                     (char *)NULL);
    return TCL_ERROR;
  }

  /* perform operation */
  mpi_call(mpi_lbadapt_grid_init, -1, level);
  mpi_lbadapt_grid_init(0, level);

  return TCL_OK;
}

int tclcommand_set_max_level(ClientData data, Tcl_Interp *interp, int argc,
                             char **argv) {
  int level;
  if (argc != 2) {
    Tcl_AppendResult(interp, "Setting a maximum refinement level requires one "
                             "parameter, specifying that level.",
                     (char *)NULL);
    return TCL_ERROR;
  }

  if (!ARG_IS_I(1, level)) {
    Tcl_AppendResult(interp,
                     "Setting l_max needs 1 parameter of type and meaning:\n",
                     (char *)NULL);
    Tcl_AppendResult(interp, "INT\n", (char *)NULL);
    Tcl_AppendResult(interp, "<max_refinement_level>\n", (char *)NULL);
    return TCL_ERROR;
  }

  /* check input for semantic correctness */
  if ((level > 18) || (level < 1)) {
    Tcl_AppendResult(interp, "allowed refinement levels are [1, 18]\n",
                     (char *)NULL);
    return TCL_ERROR;
  }

  mpi_call(mpi_lbadapt_set_max_level, -1, level);
  mpi_lbadapt_set_max_level(0, level);

  return TCL_OK;
}

int tclcommand_set_unif_ref(ClientData data, Tcl_Interp *interp, int argc,
                            char **argv) {
  /* container for parameters */
  int level = 0;

  /* check input for syntactic correctness */
  if (argc != 2) {
    Tcl_AppendResult(interp, "Setting uniform refinement requires one "
                             "parameter, specifying refinement level.",
                     (char *)NULL);
    return TCL_ERROR;
  }

  if (!ARG_IS_I(1, level)) {
    Tcl_AppendResult(
        interp, "uniform refinement needs 1 parameter of type and meaning:\n",
        (char *)NULL);
    Tcl_AppendResult(interp, "INT\n", (char *)NULL);
    Tcl_AppendResult(interp, "<refinement_level>\n", (char *)NULL);
    return TCL_ERROR;
  }

  /* check input for semantic correctness */
  if ((level > 18) || (level < 1)) {
    Tcl_AppendResult(interp, "allowed refinement levels are [1, 18]\n",
                     (char *)NULL);
    return TCL_ERROR;
  }

  mpi_call(mpi_unif_refinement, -1, level);
  mpi_unif_refinement(0, level);

  return TCL_OK;
}

int tclcommand_set_rand_ref(ClientData data, Tcl_Interp *interp, int argc,
                            char **argv) {
  int level;

  if (argc != 2) {
    Tcl_AppendResult(interp, "Setting random refinement requires one "
                             "parameter, specifying maximum refinement level.",
                     (char *)NULL);
    return TCL_ERROR;
  }

  if (!ARG_IS_I(1, level)) {
    Tcl_AppendResult(
        interp, "random refinement needs 1 parameter of type and meaning:\n",
        (char *)NULL);
    Tcl_AppendResult(interp, "INT\n", (char *)NULL);
    Tcl_AppendResult(interp, "<max_ref_level>\n", (char *)NULL);
    return TCL_ERROR;
  }

  /* check input for semantic correctness */
  if ((level > 18) || (level < 1)) {
    Tcl_AppendResult(interp, "allowed refinement levels are [1, 18]\n",
                     (char *)NULL);
    return TCL_ERROR;
  }

  mpi_call(mpi_rand_refinement, -1, level);
  mpi_rand_refinement(0, level);

  return TCL_OK;
}

int tclcommand_set_reg_ref(ClientData data, Tcl_Interp *interp, int argc,
                           char **argv) {
  if (argc == 7) {
    if (!ARG_IS_D(1, coords_for_regional_refinement[0])) {
      Tcl_AppendResult(interp, "regional refinement needs 6 parameters of"
                               "type and meaning:\n", (char*)NULL);
      Tcl_AppendResult(interp, "DBL, DBL, DBL, DBL, DBL, DBL\n", (char*)NULL);
      Tcl_AppendResult(interp, "x_min, x_max, y_min, y_max, z_min, z_max\n", (char*)NULL);
      return TCL_ERROR;
    }
    if (!ARG_IS_D(2, coords_for_regional_refinement[1])) {
      Tcl_AppendResult(interp, "regional refinement needs 6 parameters of"
                               "type and meaning:\n", (char*)NULL);
      Tcl_AppendResult(interp, "DBL, DBL, DBL, DBL, DBL, DBL\n", (char*)NULL);
      Tcl_AppendResult(interp, "x_min, x_max, y_min, y_max, z_min, z_max\n", (char*)NULL);
      return TCL_ERROR;
    }
    if (!ARG_IS_D(3, coords_for_regional_refinement[2])) {
      Tcl_AppendResult(interp, "regional refinement needs 6 parameters of"
                               "type and meaning:\n", (char*)NULL);
      Tcl_AppendResult(interp, "DBL, DBL, DBL, DBL, DBL, DBL\n", (char*)NULL);
      Tcl_AppendResult(interp, "x_min, x_max, y_min, y_max, z_min, z_max\n", (char*)NULL);
      return TCL_ERROR;
    }
    if (!ARG_IS_D(4, coords_for_regional_refinement[3])) {
      Tcl_AppendResult(interp, "regional refinement needs 6 parameters of"
                               "type and meaning:\n", (char*)NULL);
      Tcl_AppendResult(interp, "DBL, DBL, DBL, DBL, DBL, DBL\n", (char*)NULL);
      Tcl_AppendResult(interp, "x_min, x_max, y_min, y_max, z_min, z_max\n", (char*)NULL);
      return TCL_ERROR;
    }
    if (!ARG_IS_D(5, coords_for_regional_refinement[4])) {
      Tcl_AppendResult(interp, "regional refinement needs 6 parameters of"
                               "type and meaning:\n", (char*)NULL);
      Tcl_AppendResult(interp, "DBL, DBL, DBL, DBL, DBL, DBL\n", (char*)NULL);
      Tcl_AppendResult(interp, "x_min, x_max, y_min, y_max, z_min, z_max\n", (char*)NULL);
      return TCL_ERROR;
    }
    if (!ARG_IS_D(6, coords_for_regional_refinement[5])) {
      Tcl_AppendResult(interp, "regional refinement needs 6 parameters of"
                               "type and meaning:\n", (char*)NULL);
      Tcl_AppendResult(interp, "DBL, DBL, DBL, DBL, DBL, DBL\n", (char*)NULL);
      Tcl_AppendResult(interp, "x_min, x_max, y_min, y_max, z_min, z_max\n", (char*)NULL);
      return TCL_ERROR;
    }
  }
  mpi_call(mpi_bcast_parameters_for_regional_refinement, -1, 0);
  mpi_bcast_parameters_for_regional_refinement(0, 0);

  mpi_call(mpi_reg_refinement, -1, 0);
  mpi_reg_refinement(0, 0);

  return TCL_OK;
}

int tclcommand_set_geom_ref(ClientData data, Tcl_Interp *interp, int argc,
                            char **argv) {
  if (argc != 1) {
    Tcl_AppendResult(
        interp,
        "Setting geometric refinement does not allow setting parameters.",
        (char *)NULL);
    return TCL_ERROR;
  }

  mpi_call(mpi_geometric_refinement, -1, 0);
  mpi_geometric_refinement(0, 0);

  return TCL_OK;
}

int tclcommand_excl_bnd_idx_geom_ref(ClientData data, Tcl_Interp *interp,
                                     int argc, char **argv) {
  int bnd_index;

  if (argc != 2) {
    Tcl_AppendResult(interp, "Excluding a boundary from geometric refinement "
                             "requires exactly one parameter, that is the "
                             "index of the boundary to exclude.",
                     (char *)NULL);
    return TCL_ERROR;
  }

  if (!ARG_IS_I(1, bnd_index)) {
    Tcl_AppendResult(interp, "lbadapt-exclude-bnd-from-geom-ref needs 1 "
                             "parameter of type and meaning\n",
                     (char *)NULL);
    Tcl_AppendResult(interp, "INT\n", (char *)NULL);
    Tcl_AppendResult(interp, "<bnd_idx_to_ignore>\n", (char *)NULL);
    return TCL_ERROR;
  }

  /* check input for semantic correctness */
  if ((bnd_index < 0) && (n_lb_boundaries <= bnd_index)) {
    Tcl_AppendResult(interp, "boundary index to ignore must be smaller than"
                             " number of defined boundaries\n",
                     (char *)NULL);
    return TCL_ERROR;
  }

  mpi_call(mpi_exclude_boundary, -1, bnd_index);
  mpi_exclude_boundary(0, bnd_index);

  return TCL_OK;
}

#ifdef LB_ADAPTIVE_GPU
int tclcommand_gpu_show_utilization(ClientData data, Tcl_Interp *interp,
                                     int argc, char **argv) {
  int res;
  res = lbadapt_print_gpu_utilization(argv[1]);

  if (res == 0) {
    return TCL_OK;
  } else {
    return TCL_ERROR;
  }
}
#endif // LB_ADAPTIVE_GPU
#endif // LB_ADAPTIVE
