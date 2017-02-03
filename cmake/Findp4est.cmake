#
# Try to find the P4EST library
#
# This module exports:
#   P4EST_LIBRARIES
#   P4EST_INCLUDE_DIRS
#   P4EST_WITH_MPI
#   P4EST_VERSION
#   P4EST_VERSION_MAJOR
#   P4EST_VERSION_MINOR
#   P4EST_VERSION_SUBMINOR
#   P4EST_VERSION_PATCH
#

SET(P4EST_DIR "" CACHE PATH
  "An optional hint to a p4est installation/directory"
  )

#
# Search for the sc library, usually bundled with p4est. If no SC_DIR was
# given, take what we chose for p4est.
#

FIND_PATH(P4EST_INCLUDE_DIR p4est.h
  HINTS
    ${SC_DIR}
    ${P4EST_DIR}
  PATH_SUFFIXES
    sc include/p4est include src sc/src
  )

FIND_LIBRARY(SC_LIBRARIES
  NAMES sc
  HINTS
    ${SC_DIR}
    ${P4EST_DIR}
  PATH_SUFFIXES
    lib${LIB_SUFFIX} lib64 lib src sc/src
  )

FIND_PATH(SC_INCLUDE_DIR sc.h
  HINTS
    ${P4EST_DIR}
  PATH_SUFFIXES
    sc include/p4est include src sc/src
  )

FIND_LIBRARY(P4EST_LIBRARIES
  NAMES p4est
  HINTS ${P4EST_DIR}
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib src
  )

# handle the QUIETLY and REQUIRED arguments and set P4EST_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (SC DEFAULT_MSG SC_LIBRARIES SC_INCLUDE_DIR)
find_package_handle_standard_args (P4EST DEFAULT_MSG P4EST_LIBRARIES P4EST_INCLUDE_DIR)

mark_as_advanced (SC_LIBRARIES SC_INCLUDE_DIR)
mark_as_advanced (P4EST_LIBRARIES P4EST_INCLUDE_DIR)
