/*
  Copyright (C) 2010,2011,2012,2013,2014 The ESPResSo project
  Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
    Max-Planck-Institute for Polymer Research, Theory Group,

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
/** \file lb-boundaries.cpp
 *
 * Adaptive Lattice Boltzmann Scheme using CPU.
 * Implementation file for \ref lb-boundaries.hpp.
 *
 */

#include <stdlib.h>

#include "utils.hpp"
#include "constraint.hpp"
#include "communication.hpp"
#include "lb-adaptive.hpp"


#ifdef LB_ADAPTIVE

p8est_connectivity_t *conn;
p8est_t *p8est;

int refine_uniform (p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant) {
	return 1;
}

int refine_random (p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant) {
	return rand() % 2;
}

void lbadapt_init(p8est_t* p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant) {
	lbadapt_ctx_t *ctx = (lbadapt_ctx_t *) p8est->user_pointer;
	lbadapt_payload_t *data = (lbadapt_payload_t *) quadrant->p.user_data;

	data->test = 1.0;
}


#endif // LB_ADAPTIVE
