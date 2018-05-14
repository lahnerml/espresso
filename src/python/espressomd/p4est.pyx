#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, absolute_import, division
include "myconfig.pxi"
import os
import cython
import numpy as np
cimport numpy as np
from .actors cimport Actor
from . cimport cuda_init
from . import cuda_init
from globals cimport *
from copy import deepcopy
from . import utils
from espressomd.utils import array_locked, is_valid_type
from libcpp.string cimport string

# Actor class
####################################################
cdef class GridInteraction(Actor):
    def _p4est_init(self):
        raise Exception(
            "Subclasses of HydrodynamicInteraction must define the _lb_init() method.")

# LBFluid main class
####################################################
IF LB_ADAPTIVE or EK_ADAPTIVE or ES_ADAPTIVE:
    cdef class P4est(GridInteraction):
        """
        Initialize p4est grid interaction routines

        """

        # validate the given parameters on actor initialization
        ####################################################
        def validate_params(self):
            default_params = self.default_params()

        # list of valid keys for parameters
        ####################################################
        def valid_keys(self):
            return "min_ref_level", "max_ref_level", \
                   "partitioning", \
                   "threshold_velocity", "threshold_vorticity"

        # list of esential keys required for the fluid
        ####################################################
        def required_keys(self):
            return []

        # list of default parameters
        ####################################################
        def default_params(self):
            return {"min_ref_level": -1,
                    "max_ref_level": -1,
                    "partitioning" : "n_cells",
                    "threshold_velocity": [0.0, 1.0],
                    "threshold_vorticity": [0.0, 1.0]}

        # function that calls wrapper functions which set the parameters at C-Level
        ####################################################
        def _set_params_in_es_core(self):
            default_params = self.default_params()

            if python_p4est_set_min_level(self._params["min_ref_level"]):
                raise Exception("p4est_utils_set_min_level error")

            if python_p4est_set_max_level(self._params["max_ref_level"]):
                raise Exception("p4est_utils_set_max_level error")

            if python_p4est_set_partitioning(self._params["partitioning"]):
                raise Exception("p4est_utils_set_partitioning error")

            if python_p4est_set_threshold_velocity(self._params["threshold_velocity"]):
                raise Exception("p4est_utils_set_threshold_velocity error")

            if python_p4est_set_threshold_vorticity(self._params["threshold_vorticity"]):
                raise Exception("p4est_utils_set_threshold_vorticity error")
            utils.handle_errors("p4est activation")

        # function that calls wrapper functions which get the parameters from C-Level
        ####################################################
        def _get_params_from_es_core(self):
            default_params = self.default_params()

            if not self._params["min_ref_level"] == default_params["min_ref_level"]:
                if python_p4est_get_min_level(self._params["min_ref_level"]):
                    raise Exception("p4est_utils_get_min_level error")

            if not self._params["max_ref_level"] == default_params["max_ref_level"]:
                if python_p4est_get_max_level(self._params["max_ref_level"]):
                    raise Exception("p4est_utils_get_max_level error")

            if not self._params["partitioning"] == default_params["partitioning"]:
                if python_p4est_get_partitioning(self._params["partitioning"]):
                    raise Exception("p4est_utils_get_partitioning error")

            if not self._params["threshold_velocity"] == default_params["threshold_velocity"]:
                if python_p4est_get_threshold_velocity(self._params["threshold_velocity"]):
                    raise Exception("p4est_utils_get_threshold_velocity error")

            if not self._params["threshold_vorticity"] == default_params["threshold_vorticity"]:
                if python_p4est_get_threshold_vorticity(self._params["threshold_vorticity"]):
                    raise Exception("p4est_utils_get_threshold_vorticity error")

            return self._params

        # Activate Actor
        ####################################################
        def _activate_method(self):
            self.validate_params()
            self._set_params_in_es_core()
            IF EK_ADAPTIVE or ES_ADAPTIVE or LB_ADAPTIVE:
                return
            ELSE:
                raise Exception("No adaptive components are compiled in.")

        def _deactivate_method(self):
            pass

        # Adapt Grid
        ####################################################
        def uniform_refinement(self, p_ref_steps):
            cdef int c_ref_steps = p_ref_steps
            if (p4est_utils_uniform_refinement(c_ref_steps)):
                raise Exception("p4est_utils_uniform_refinement error")

            return 0

        ####################################################

        def random_refinement(self, p_ref_steps):
            cdef int c_ref_steps = p_ref_steps
            if (p4est_utils_random_refinement(c_ref_steps)):
                raise Exception("p4est_utils_random_refinement error")

            return 0

        ####################################################

        def regional_coarsening(self, p_bb_coords):
            cdef double c_bb_coords[6]
            c_bb_coords = p_bb_coords
            if (p4est_utils_regional_coarsening(c_bb_coords)):
                raise Exception("p4est_utils_regional_coarsening error")

            return 0

        ####################################################

        def regional_refinement(self, p_bb_coords):
            cdef double c_bb_coords[6]
            c_bb_coords = p_bb_coords
            if (p4est_utils_regional_refinement(c_bb_coords)):
                raise Exception("p4est_utils_regional_refinement error")

            return 0

        ####################################################

        def geometric_refinement(self):
            if (p4est_utils_geometric_refinement()):
                raise Exception("p4est_utils_geometric_refinement error")

            return 0

        ####################################################

        def inverse_geometric_refinement(self):
            if (p4est_utils_inverse_geometric_refinement()):
                raise Exception("p4est_utils_inverse_geometric_refinement error")

            return 0

        ####################################################

        def adapt_grid(self):
            if (p4est_utils_adapt_grid()):
                raise Exception("p4est_utils_adapt_grid error")

            return 0

        ####################################################

        def set_refinement_area(self, p_bb_coords, p_vel):
            cdef double c_bb_coords[6]
            c_bb_coords = p_bb_coords
            cdef double c_vel[3]
            c_vel = p_vel
            if (p4est_utils_set_refinement_area(c_bb_coords, c_vel)):
                raise Exception("p4est_utils_set_refinement_area error")

            return 0
