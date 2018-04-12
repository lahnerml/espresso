#ifndef P4EST_UTILS_HPP
#define P4EST_UTILS_HPP

#include "grid.hpp"
#include "p4est_dd.hpp"
#include "utils.hpp"

#if (defined(LB_ADAPTIVE) || defined(DD_P4EST))

#include <memory>
#include <p4est_to_p8est.h>
#include <p8est.h>
#include <p8est_connectivity.h>
#include <p8est_ghost.h>
#include <p8est_mesh.h>
#include <p8est_meshiter.h>
#include <p8est_virtual.h>
#include <stdint.h>
#include <vector>

/*****************************************************************************/
/** \name Generic helper functions                                           */
/*****************************************************************************/

namespace std
{
  template<>
  struct default_delete<p4est_t>
  {
    void operator()(p4est_t *p) const { if (p != nullptr) p4est_destroy(p); }
  };
  template<>
  struct default_delete<p4est_ghost_t>
  {
    void operator()(p4est_ghost_t *p) const { if (p != nullptr) p4est_ghost_destroy(p); }
  };
  template<>
  struct default_delete<p4est_mesh_t>
  {
    void operator()(p4est_mesh_t *p) const { if (p != nullptr) p4est_mesh_destroy(p); }
  };
  template<>
  struct default_delete<p4est_virtual_t>
  {
    void operator()(p4est_virtual_t *p) const { if (p != nullptr) p4est_virtual_destroy(p); }
  };
  template<>
  struct default_delete<p4est_virtual_ghost_t>
  {
    void operator()(p4est_virtual_ghost_t *p) const { if (p != nullptr) p4est_virtual_ghost_destroy(p); }
  };
  template<>
  struct default_delete<p4est_connectivity_t>
  {
    void operator()(p4est_connectivity_t *p) const { if (p != nullptr) p4est_connectivity_destroy(p); }
  };
  template<>
  struct default_delete<p4est_meshiter_t>
  {
      void operator()(p4est_meshiter_t *p) const { if (p != nullptr) p4est_meshiter_destroy(p); }
  };
  template<>
  struct default_delete<sc_array_t>
  {
      void operator()(sc_array_t *p) const { if (p != nullptr) sc_array_destroy(p); }
  };
}

// Don't use it. Can lead to nasty bugs.
template <typename T>
struct castable_unique_ptr: public std::unique_ptr<T> {
  using Base = std::unique_ptr<T>;
  constexpr castable_unique_ptr(): Base() {}
  constexpr castable_unique_ptr(std::nullptr_t n): Base(n) {}
  castable_unique_ptr(T* p): Base(p) {}
  castable_unique_ptr(Base&& other): Base(std::move(other)) {}
  operator T*() const { return this->get(); }
  operator void *() const { return this->get(); }
};

#ifdef LB_ADAPTIVE
extern int lb_conn_brick[3];
// relative threshold values for refinement and coarsening.
extern double vel_thresh[2];
extern double vort_thresh[2];
#endif // LB_ADAPTIVE

enum class forest_order {
#ifdef DD_P4EST
  short_range = 0,
#endif // DD_P4EST
#ifdef LB_ADAPTIVE
  adaptive_LB
#endif // LB_ADAPTIVE
};

struct p4est_utils_forest_info_t {
  p4est_t *p4est;
  std::vector<int> p4est_space_idx;
  int coarsest_level_local;
  int coarsest_level_ghost;
  int coarsest_level_global;
  int finest_level_local;
  int finest_level_global;
  int finest_level_ghost;

  p4est_utils_forest_info_t(p4est_t *p4est)
      : p4est(p4est), p4est_space_idx(),
        coarsest_level_local(P8EST_QMAXLEVEL),
        coarsest_level_ghost(P8EST_QMAXLEVEL),
        coarsest_level_global(P8EST_QMAXLEVEL), finest_level_local(-1),
        finest_level_global(-1), finest_level_ghost(-1) {}
};

/** Returns a const reference to the forest_info of "fo".
 * Throws a std::out_of_range if the forest_info of "fo" has not been created
 * yet.
 *
 * @param fo specifier which forest_info to return
 */
const p4est_utils_forest_info_t &p4est_utils_get_forest_info(forest_order fo);

/** For algorithms like mapping a position to a quadrant to work we need a
 * synchronized version of the quadrant offsets of each tree.
 *
 * @param p4ests     List of all p4ests in the current simulation
 */
void p4est_utils_prepare(std::vector<p8est_t *> p4ests);

/**
 * Destroy and rebuild forest_info, partition the forests, and recreate
 * p4est_ghost, p4est_mesh, p4est_virtual, and p4est_virtual_ghost.
 * This function leaves payload untouched and is most helpful for static grid
 * refinement before the simulation has started.
 *
 * @param btype      Neighbor information that is to be included
 */
void p4est_utils_rebuild_p4est_structs(p4est_connect_type_t btype);

/*****************************************************************************/
/** \name Geometric helper functions                                         */
/*****************************************************************************/
/*@{*/
/** Transform coordinates in-place from p4est-tree coordinates (a tree is a
 * unit-cube) to simulation domain.
 *
 * @param x    Position that is transformed in place.
 */
inline void tree_to_boxlcoords(double x[3]) {
  for (int i = 0; i < P8EST_DIM; ++i) {
#if defined(DD_P4EST) && defined(LB_ADAPTIVE)
    if (dd_p4est_get_p4est() != nullptr) {
      x[i] *= (box_l[i] / dd_p4est_get_n_trees(i));
    } else {
      x[i] *= (box_l[i] / lb_conn_brick[i]);
    }
#elif defined(DD_P4EST)
    x[i] *= (box_l[i] / dd_p4est_get_n_trees(i));
#elif defined(LB_ADAPTIVE)
    x[i] *= (box_l[i] / lb_conn_brick[i]);
#endif
  }
#if !defined(DD_P4EST) && !defined(LB_ADAPTIVE)
  fprintf(stderr, "Function tree_to_boxlcoords only works if DD_P4EST or "
                  "LB_ADAPTIVE is activated.\n");
  errexit();
#endif
}

/** Transform coordinates in-place from simulation domain to p4est
 * tree-coordinates (a tree is a unit-cube).
 *
 * @param x    Position that is transformed in place.
 */
inline void boxl_to_treecoords(double x[3]) {
  for (int i = 0; i < P8EST_DIM; ++i) {
#if defined(DD_P4EST) && defined(LB_ADAPTIVE)
    if (dd_p4est_get_p4est() != nullptr) {
      x[i] /= (box_l[i] / dd_p4est_get_n_trees(i));
    } else {
      x[i] /= (box_l[i] / lb_conn_brick[i]);
    }
#elif defined(DD_P4EST)
  x[i] /= (box_l[i] / dd_p4est_get_n_trees(i));
#elif defined(LB_ADAPTIVE)
  x[i] /= (box_l[i] / lb_conn_brick[i]);
#endif
  }
#if !defined(DD_P4EST) && !defined(LB_ADAPTIVE)
  fprintf(stderr, "Function tree_to_boxlcoords only works if DD_P4EST or "
                  "LB_ADAPTIVE is activated.\n");
  errexit();
#endif
}


/** Get the coordinates of the front lower left corner of a quadrant.
 *
 * \param [in]  p8est    the forest
 * \param [in]  which_tree the tree in the forest containing \a q
 * \param [in]  q      the quadrant
 * \param [out] xyz    the coordinates of the midpoint of \a q
 */
void p4est_utils_get_front_lower_left(p8est_t *p8est,
                                      p4est_topidx_t which_tree,
                                      p8est_quadrant_t *q, double *xyz);

/** Get the coordinates of the front lower left corner of a quadrant
 *
 * \param [in]  mesh_iter  A mesh-based iterator.
 * \param [out] xyz        The coordinates of the the front lower left corner
 *                         of the current quadrant that mesh_iter is pointing
 *                         to.
 */

void p4est_utils_get_front_lower_left(p8est_meshiter_t *mesh_iter,
                                      double *xyz);
/** Get the coordinates of the midpoint of a quadrant.
 *
 * \param [in]  p8est    the forest
 * \param [in]  which_tree the tree in the forest containing \a q
 * \param [in]  q      the quadrant
 * \param [out] xyz    the coordinates of the midpoint of \a q
 */
void p4est_utils_get_midpoint(p8est_t *p8est, p4est_topidx_t which_tree,
                              p8est_quadrant_t *q, double xyz[3]);

/** Get the coordinates of the midpoint of a quadrant
 *
 * \param [in]  mesh_iter  A mesh-based iterator.
 * \param [out] xyz        The coordinates of the the midpoint of the current
 *                         quadrant that mesh_iter is pointing to.
 */
void p4est_utils_get_midpoint(p8est_meshiter_t *mesh_iter, double *xyz);

/** Obtain a Morton-index by interleaving 3 integer coordinates.
 *
 * @param x, y, z  The coordinates
 * @return         Morton-index of given position.
 */
int64_t p4est_utils_cell_morton_idx(int x, int y, int z);

/** Obtain a Morton-index from a quadrant and its treeid.
 *
 * @param q        Quadrant of a given tree.
 * @param tree     Tree-id holding given quadrant.
 * @return         Morton-index of given quadrant.
 */
int64_t p4est_utils_global_idx(p8est_quadrant_t *q, p4est_topidx_t tree);

/** Map a process-local Morton-index obtained from
 * \ref p4est_utils_cell_morton_idx to a local qid.
 *
 * @param forest   The position of the forest info containing the p4est for
 *                 which to map.
 * @param idx      The Morton index as calculated by
 *                 \ref p4est_utils_cell_morton_idx
 * @return         The quadrant index at the given index.
 */
p4est_locidx_t p4est_utils_idx_to_qid(forest_order forest, p4est_gloidx_t idx);

/** Map a geometric position to the respective processor.
 *
 * @param forest   The position of the forest info containing the p4est for
 *                 which to map.
 * @param xyz      The position to map.
 * @return         The MPI rank which holds a quadrant at the given position for
 *                 the current p4est.
 */
int p4est_utils_pos_to_proc(forest_order forest, double* xyz);

/** Map an index to the respective processor.
 *
 * @param forest   The forest info containing the p4est for which to map.
 * @param idx      The Morton index as calculated by
 *                 \ref p4est_utils_cell_morton_idx.
 * @return         The MPI rank which holds a quadrant at the given position for
 *                 the current p4est.
 */
int p4est_utils_idx_to_proc(forest_order forest, p4est_gloidx_t idx);
/*@}*/

/*****************************************************************************/
/** \name Mapping directions                                                 */
/*****************************************************************************/

/*@{*/
/** Mapping between ESPResSo's definition of c_i in LBM to p4est directions */
// clang-format off
const int ci_to_p4est[18] = {  1,  0,  3,  2,  5,  4, 17, 14, 15,
                              16, 13, 10, 11, 12,  9,  6,  7,  8 };

const int p4est_to_ci[18] = {  2,  1,  4,  3,  6,  5, 16, 17, 18,
                              15, 12, 13, 14, 11,  8,  9, 10,  7 };
// clang-format on
/*@}*/

/*****************************************************************************/
/** \name Data Allocation and Deallocation                                   */
/*****************************************************************************/
/*@{*/
template <typename T>
/** Generic function to allocate level-wise data-structure with potentially
 * empty lists from 0 to P8EST_QMAXLEVEL.
 *
 * @param T           Type of numerical payload.
 * @param data        Pointer to payload struct
 * @param mesh        Mesh of current p4est.
 * @param local_data  Bool indicating if local or ghost information is relevant.
 */
int p4est_utils_allocate_levelwise_storage(std::vector<std::vector<T>> &data,
                                           p4est_mesh_t *mesh,
                                           p4est_virtual_t *virtual_quads,
                                           bool local_data) {
  P4EST_ASSERT(data.empty());

  // allocate data for each level
  data = std::vector<std::vector<T>>(P8EST_QMAXLEVEL, std::vector<T>());
  P4EST_ASSERT(data.size() == P8EST_QMAXLEVEL);

  int quads_on_level;

  for (size_t level = 0; level < P8EST_QMAXLEVEL; ++level) {
    quads_on_level =
        local_data
            ? (mesh->quad_level + level)->elem_count +
                  P8EST_CHILDREN * (virtual_quads->virtual_qlevels + level)->elem_count
            : (mesh->ghost_level + level)->elem_count +
                  P8EST_CHILDREN * (virtual_quads->virtual_glevels + level)->elem_count;
    data[level] = std::vector<T>(quads_on_level);
    P4EST_ASSERT(data[level].size() == (size_t) quads_on_level);
  }

  return 0;
}

template <typename T>
/** Deallocate a level-wise data-structure with potentially empty lists of
 * numerical payload from level 0 .. P8EST_QMAXLEVEL.
 *
 * @param T           Data-type of numerical payload.
 * @param data        Pointer to payload struct
 */
int p4est_utils_deallocate_levelwise_storage(
    std::vector<std::vector<T>> &data) {
  if (!data.empty()) {
    for (int level = 0; level < P8EST_QMAXLEVEL; ++level) {
      data[level].clear();
    }
  }
  data.clear();

  return 0;
}

template <typename T>
int prepare_ghost_exchange(std::vector<std::vector<T>> &local_data,
                           std::vector<T *> &local_pointer,
                           std::vector<std::vector<T>> &ghost_data,
                           std::vector<T *> &ghost_pointer) {
  P4EST_ASSERT(ghost_data.size() == 0 ||
               ghost_data.size() == local_data.size());
  for (unsigned int i = 0; i < local_data.size(); ++i) {
    local_pointer[i] = local_data[i].data();
    if (ghost_data.size() != 0) {
      ghost_pointer[i] = ghost_data[i].data();
    }
  }
  return 0;
}
/*@}*/

/*****************************************************************************/
/** \name Grid Change                                                        */
/*****************************************************************************/
/*@{*/
/** Function that handles grid alteration. After calling this function the grid
 * has changed and everything is set to perform the next integration step.
 */
int p4est_utils_adapt_grid();

template <typename T>
/** Skeleton for copying data.
 *
 * @param T           Data-type of numerical payload.
 * @param p4est_old   p4est before grid change
 * @param p4est_new   p4est after grid change
 * @param quad_old    Current quad in old p4est
 * @param quad_new    Current quad in new p4est
 * @param which_tree  Current tree
 * @param data_old    Numerical payload of old quadrant from which \a data_new
 *                    will be generated.
 * @param data_new    Numerical payload of new quadrant.
 * @return int
 */
int data_transfer(p8est_t *p4est_old, p8est_t *p4est_new,
                  p8est_quadrant_t *quad_old, p8est_quadrant_t *quad_new,
                  int which_tree, T *data_old, T *data_new);

template <typename T>
/** Skeleton for restricting data.
 *
 * @param T           Data-type of numerical payload.
 * @param p4est_old   p4est before grid change
 * @param p4est_new   p4est after grid change
 * @param quad_old    Current quad in old p4est
 * @param quad_new    Current quad in new p4est
 * @param which_tree  Current tree
 * @param data_old    Numerical payload of old quadrant from which \a data_new
 *                    will be generated.
 * @param data_new    Numerical payload of new quadrant.
 * @return int
 */
int data_restriction(p8est_t *p4est_old, p8est_t *p4est_new,
                     p8est_quadrant_t *quad_old, p8est_quadrant_t *quad_new,
                     int which_tree, T *data_old, T *data_new);

template <typename T>
/** Skeleton for interpolating data.
 *
 * @param T           Data-type of numerical payload.
 * @param p4est_old   p4est before grid change
 * @param p4est_new   p4est after grid change
 * @param quad_old    Current quad in old p4est
 * @param quad_new    Current quad in new p4est
 * @param which_tree  Current tree
 * @param data_old    Numerical payload of old quadrant from which \a data_new
 *                    will be generated.
 * @param data_new    Numerical payload of new quadrant.
 * @return int
 */
int data_interpolation(p8est_t *p4est_old, p8est_t *p4est_new,
                       p8est_quadrant_t *quad_old, p8est_quadrant_t *quad_new,
                       int which_tree, T *data_old, T *data_new);

template <typename T>
/** Generic mapping function from custom-managed level-wise data to a temporary
 * flat data storage.
 *
 * @param T           Data-type of numerical payload.
 * @param p4est_old   p4est before refining, coarsening, and balancing.
 * @param mesh_old    Lookup tables for \a p4est_old.
 * @param p4est_new   p4est after refining, coarsening, and balancing.
 * @param local_data_levelwise   Level-wise numerical payload, corresponds to
 *                               \a p4est_old
 * @param mapped_data_flat  Already allocated container for temporarily storing
 *                          mapped data from old to new grid before re-
 *                          establishing a good load-balancing. corresponding
 *                          to \a p4est_new.
 *                          CAUTION: Needs to be allocated and all numerical
 *                                   payload is supposed to be filled with 0.
 */
int p4est_utils_post_gridadapt_map_data(
    p4est_t *p4est_old, p4est_mesh_t *mesh_old,
    p4est_virtual_t *virtual_quads, p4est_t *p4est_new,
    std::vector<std::vector<T>> &local_data_levelwise, T *mapped_data_flat);

template <typename T>
/** Generic function to re-establish a proper load balacing after the grid has
 * changed. Transfer data to the respective processors according to new
 * partition table.
 *
 * @param T           Data-type of numerical payload.
 * @param p4est_old   p4est before calling p4est_partition.
 * @param p4est_new   p4est after calling p4est_partition.
 * @param data_mapped Mapped data, result of calling post_gridadapt_map_data.
 * @param data_partitioned  Partitioned data, sorted by rank of origin. Vector
 *                          needs to be MPI_Size long, individual subvectors
 *                          will be allocated by the function and they will be
 *                          used as receive buffers.
 * @return int
 */
int p4est_utils_post_gridadapt_data_partition_transfer(
    p8est_t *p4est_old, p8est_t *p4est_new, T *data_mapped,
    std::vector<std::vector<T>> &data_partitioned);

template <typename T>
/** After all local data has been received insert in newly allocated level-wise
 * data structure.
 *
 * @param T           Data-type of numerical payload.
 * @param p4est_new   p4est after partitioning.
 * @param mesh_new    Lookup tables referring to \a p4est_new
 * @param data_partitioned  Partitioned data, sorted by rank of origin. Vector
 *                          needs to be MPI_Size long.
 * @param data_levelwise  Level-wise numerical payload.
 * @return int
 */
int p4est_utils_post_gridadapt_insert_data(
    p4est_t *p4est_new, p4est_mesh_t *mesh_new, p4est_virtual_t *virtual_quads,
    std::vector<std::vector<T>> &data_partitioned,
    std::vector<std::vector<T>> &data_levelwise);
/*@}*/

/*****************************************************************************/
/** \name Partition different p4ests                                         */
/*****************************************************************************/
/*@{*/
/** Partition two different p4ests such that their partition boundaries match.
 *  In case the reference is coarser or equal, the process boundaries are identical.
 *  Otherwise, only finer cells match the process for the reference.
 *
 * @param[in]  p4est_ref   p4est with reference partition boundaries.
 * @param[out] p4est_mod   p4est to be modified such that its boundaries match
 *                         those of \a p4est_ref
 */
void p4est_utils_partition_multiple_forests(forest_order reference,
                                            forest_order modify);
/*@}*/

/** Returns true if the process boundaries of two p4ests are aligned.
 *
 * @param[in] t1    First tree
 * @param[in] t2    Second tree
 */
bool p4est_utils_check_alignment(const p4est_t *t1, const p4est_t *t2);

/** Computes the finest common tree out of two given p4est trees on the same
 * connectivity.
 * Requires all finer cells in t2 to match the process of t1. (see
 * \ref p4est_utils_partition_multiple_forests)
 *
 * @param[in] t1  reference tree for FCT
 * @param[in] t2  base tree for FCT. this tree is copied and coarsened.
 *
 */
p4est_t *p4est_utils_create_fct(p4est_t *t1, p4est_t *t2);

/** Repartitions t1 and t2 wrt. the FCT. The respective weights are combined
 * with the factors a1 and a2.
 * It is required for t1 and t2 to have identical process boundaries.
 *
 * @param[inout] t1   First tree
 * @param[in]    w1   Weights for t1
 * @param[in]    a1   Weight factor for w1
 * @param[inout] t2   Second tree
 * @param[in]    w2   Weights for t2
 * @param[in]    a2   Weight factor for w2
 *
 */
void p4est_utils_weighted_partition(p4est_t *t1, const std::vector<double> &w1,
                                    double a1, p4est_t *t2,
                                    const std::vector<double> &w2, double a2);

#endif // defined (LB_ADAPTIVE) || defined (DD_P4EST)
#endif // P4EST_UTILS_HPP
