include "myconfig.pxi"

IF CATALYTIC_REACTIONS:
    cdef extern from "reaction.hpp":
        void reactions_sanity_checks()
        void local_setup_reaction()
        void integrate_reaction()

    cdef extern from "communication.hpp":
        void mpi_setup_reaction()
