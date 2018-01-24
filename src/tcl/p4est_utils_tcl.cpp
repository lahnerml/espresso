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

#include "communication.hpp"
#include "p4est_utils_tcl.hpp"


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
#endif // (defined(LB_ADAPTIVE) || defined(DD_P4EST)
