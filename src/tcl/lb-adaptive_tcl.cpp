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

#include <stdio.h>


int tclcommand_setup_grid(ClientData data, Tcl_Interp *interp, int argc, char **argv) {
	/* verify input */
	if (argc != 1)
	{
		Tcl_AppendResult(interp, "Initial grid setup is parameter free.\n", (char *)NULL);
    return TCL_ERROR;
	}

	/* perform operation */
	setup_grid();

	return TCL_OK;
}

int tclcommand_set_unif_ref(ClientData data, Tcl_Interp *interp, int argc, char **argv) {
	/* container for parameters */
	int level;

	/* check input for syntactic correctness */
  if (argc != 2)
  {
		Tcl_AppendResult(interp,
				"Setting uniform refinement requires one parameter, specifying refinement level.",
				(char *)NULL);
    return TCL_ERROR;
  }

  if(! ARG_IS_I(1, level)) {
    Tcl_AppendResult(interp, "uniform refinement needs 1 parameter of type and meaning:\n", (char *)NULL);
    Tcl_AppendResult(interp, "INT\n", (char *)NULL);
    Tcl_AppendResult(interp, "<refinement_level>\n", (char *)NULL);
    return TCL_ERROR;
  }

  /* check input for semantic correctness */
  if ((level > 18) || (level < 1))
  {
    Tcl_AppendResult(interp, "allowed refinement levels are [1, 18]\n", (char *)NULL);
    return TCL_ERROR;
  }

  mpi_call(mpi_unif_refinement, 1, level);
  unif_refinement(level);

  return TCL_OK;
}

int tclcommand_set_rand_ref(ClientData data,Tcl_Interp *interp, int argc, char **argv) {
  int level;

	if (argc != 2)
  {
		Tcl_AppendResult(interp,
				"Setting random refinement requires one parameter, specifying maximum refinement level.",
				(char *)NULL);
    return TCL_ERROR;
  }

  if(! ARG_IS_I(1, level)) {
    Tcl_AppendResult(interp, "random refinement needs 1 parameter of type and meaning:\n", (char *)NULL);
    Tcl_AppendResult(interp, "INT\n", (char *)NULL);
    Tcl_AppendResult(interp, "<max_ref_level>\n", (char *)NULL);
    return TCL_ERROR;
  }

  /* check input for semantic correctness */
  if ((level > 18) || (level < 1))
  {
    Tcl_AppendResult(interp, "allowed refinement levels are [1, 18]\n", (char *)NULL);
    return TCL_ERROR;
  }

  mpi_call(mpi_rand_refinement, -1, level);
  rand_refinement(level);

  return TCL_OK;
}
