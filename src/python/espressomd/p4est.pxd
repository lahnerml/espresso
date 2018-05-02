#
# Copyright (C) 2013,2014,2015,2016 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import print_function, absolute_import
include "myconfig.pxi"
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from .actors cimport Actor

cdef class GridInteraction(Actor):
    pass

IF LB_ADAPTIVE or EK_ADAPTIVE or ES_ADAPTIVE:

    ##############################################
    #
    # extern functions and structs
    #
    ##############################################

    cdef extern from "p4est_utils.hpp":

        ##############################################
        #
        # Python p4est struct clone of C-struct
        #
        ##############################################
        ctypedef struct p4est_parameters:
            int min_ref_level
            int max_ref_level
            string partitioning
            double threshold_velocity[2]
            double threshold_vorticity[2]

        ##############################################
        #
        # init struct
        #
        ##############################################
        ctypedef struct p4est_parameters:
            p4est_parameters p4est_params


        ##############################################
        #
        # exported C-functions from p4est_utils.hpp
        #
        ##############################################
        int p4est_utils_set_min_level(int c_lvl)
        int p4est_utils_get_min_level(int *c_lvl)
        int p4est_utils_set_max_level(int c_lvl)
        int p4est_utils_get_max_level(int *c_lvl)
        int p4est_utils_set_partitioning(string c_part)
        int p4est_utils_get_partitioning(string *c_part)
        int p4est_utils_set_threshold_velocity(double *c_thresh)
        int p4est_utils_get_threshold_velocity(double *c_thresh)
        int p4est_utils_set_threshold_vorticity(double *c_thresh)
        int p4est_utils_get_threshold_vorticity(double *c_thresh)
        int p4est_utils_uniform_refinement(int ref_steps)
        int p4est_utils_random_refinement(int ref_steps)
        int p4est_utils_regional_coarsening(double *c_bbcoords)
        int p4est_utils_regional_refinement(double *c_bbcoords)
        int p4est_utils_geometric_refinement_exclude_boundary_index(int index)
        int p4est_utils_geometric_refinement()
        int p4est_utils_inverse_geometric_refinement()
        int p4est_utils_adapt_grid()


    ###############################################
    #
    # Wrapper-functions for access to C-pointer: Set params
    #
    ###############################################
    cdef inline python_p4est_set_min_level(p_lvl):

        cdef int c_lvl
        # get pointers
        c_lvl = p_lvl
        # call c-function
        if(p4est_utils_set_min_level(c_lvl)):
            raise Exception("p4est_utils_set_min_level error at C-level"
                            " interface")

        return 0

    ###############################################

    cdef inline python_p4est_set_max_level(p_lvl):

        cdef int c_lvl
        # get pointers
        c_lvl = p_lvl
        # call c-function
        if(p4est_utils_set_max_level(c_lvl)):
            raise Exception("p4est_utils_set_max_level error at C-level"
                            " interface")

        return 0

    ###############################################

    cdef inline python_p4est_set_partitioning(p_part):
        cdef string c_part
        c_part = p_part.encode("UTF-8")
        if(p4est_utils_set_partitioning(c_part)):
            raise Exception("p4est_utils_set_partitioning errot at C-level"
                            " interface")
        return 0

    ###############################################

    cdef inline python_p4est_set_threshold_velocity(p_thresh):

        cdef double c_thresh[2]
        # get pointers
        c_thresh = p_thresh
        # call c-function
        if(p4est_utils_set_threshold_velocity(c_thresh)):
            raise Exception("p4est_utils_set_threshold_velocity error at C-level interface")

        return 0

    ###############################################

    cdef inline python_p4est_set_threshold_vorticity(p_thresh):

        cdef double c_thresh[2]
        # get pointers
        c_thresh = p_thresh
        # call c-function
        if(p4est_utils_set_threshold_vorticity(c_thresh)):
            raise Exception("p4est_utils_set_threshold_vorticity error at C-level interface")

        return 0

    ###############################################


    ###############################################
    #
    # Wrapper-functions for access to C-pointer: Get params
    #
    ###############################################
    cdef inline python_p4est_get_min_level(p_lvl):

        cdef int c_lvl[1]
        # call c-function
        if(p4est_utils_get_min_level(c_lvl)):
            raise Exception("p4est_utils_get_min_level error at C-level interface")
        if(isinstance(c_lvl, float)):
            p_lvl = <int > c_lvl[0]
        else:
            p_lvl = c_lvl

        return 0

    ###############################################

    cdef inline python_p4est_get_max_level(p_lvl):

        cdef int c_lvl[1]
        # call c-function
        if(p4est_utils_get_max_level(c_lvl)):
            raise Exception("p4est_utils_get_max_level error at C-level interface")
        if(isinstance(c_lvl, float)):
            p_lvl = <int > c_lvl[0]
        else:
            p_lvl = c_lvl

        return 0

    ###############################################

    cdef inline python_p4est_get_threshold_velocity(p_thresh):

        cdef double c_thresh[2]
        if (p4est_utils_get_threshold_velocity(c_thresh)):
            raise Exception("p4est_utils_get_threshold_velocity error at C-level interface")
        p_thresh = c_thresh

        return 0

    ###############################################

    cdef inline python_p4est_get_threshold_vorticity(p_thresh):

        cdef double c_thresh[2]
        if (p4est_utils_get_threshold_vorticity(c_thresh)):
            raise Exception("p4est_utils_get_threshold_vorticity error at C-level interface")
        p_thresh = c_thresh

        return 0

    ###############################################
