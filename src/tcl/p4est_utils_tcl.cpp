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

#include "p4est_utils_tcl.hpp"

#include "communication.hpp"
#include "p4est_utils.hpp"

#if (defined(LB_ADAPTIVE) || defined(DD_P4EST))
int tclcommand_adapt_grid(ClientData data, Tcl_Interp *interp, int argc,
                          char **argv) {
  if (argc != 1) {
    Tcl_AppendResult(interp, "Calling dynamic adaptivity does not allow "
                             "setting parameters.", (char *)NULL);
    return TCL_ERROR;
  }
  mpi_call(mpi_adapt_grid, -1, -1);
  mpi_adapt_grid(0, -1);

  return 0;
}

void print_usage_set_thresh(ClientData data, Tcl_Interp *interp) {
  Tcl_AppendResult(interp, "Setting refinement criteria requires 3 "
                           "parameters:\n[criterion] [thresh_coarsening] "
                           "[thresh_refine]\n");
}

int tclcommand_set_adapt_thresh(ClientData data, Tcl_Interp *interp, int argc,
                                char **argv) {
  if(argc != 4) {
    print_usage_set_thresh(data, interp);
    return TCL_ERROR;
  }

  argc--; argv++;

  if(ARG0_IS_S_EXACT("velocity")) {
    --argc; ++argv;
    if(!ARG0_IS_D(vel_thresh[0]) || !ARG1_IS_D(vel_thresh[1])) {
      print_usage_set_thresh(data, interp);
      return TCL_ERROR;
    }
    mpi_call(mpi_bcast_thresh_vel, -1, 0);
    mpi_bcast_thresh_vel(0, 0);
  }
  else if (ARG0_IS_S_EXACT("vorticity")) {
    --argc; ++argv;
    if(!ARG0_IS_D(vort_thresh[0]) || !ARG1_IS_D(vort_thresh[1])) {
      print_usage_set_thresh(data, interp);
      return TCL_ERROR;
    }
    mpi_call(mpi_bcast_thresh_vort, -1, 0);
    mpi_bcast_thresh_vort(0, 0);
  }
  else {
    print_usage_set_thresh(data, interp);
    return TCL_ERROR;
  }

  return TCL_OK;
}
#endif // (defined(LB_ADAPTIVE) || defined(DD_P4EST)
