/*
  Copyright (C) 2010,2011,2012,2013,2014,2015,2016 The ESPResSo project
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

#include "p4est_dd.hpp"
#include "p4est_dd_tcl.hpp"
#include "communication.hpp"

#define LEN(x) (sizeof(x) / sizeof(*(x)))


int tclcommand_md_vtk(ClientData data, Tcl_Interp *interp,
                     int argc, char *argv[])
{
  char *filename;
  
  if (argc < 2) {
    Tcl_AppendResult(interp, "wrong # args:  should be \"",
                     argv[0], " <filename> \"",
                     (char *) NULL);
    return (TCL_ERROR);
  }
  filename = argv[1];
  
   /* this is parallel io, i.e. we have to communicate the filename to all
   * other processes. */
  int len = strlen(filename) + 1;

  /* call mpi printing routine on all slaves and communicate the filename */
  mpi_call(mpi_dd_p4est_write_particle_vtk, -1, len);
  MPI_Bcast(filename, len, MPI_CHAR, 0, comm_cart);
  
  dd_p4est_write_particle_vtk(filename);

  return (TCL_OK);
}
