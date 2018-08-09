#ifndef P4EST_UTILS_HPP
#define P4EST_UTILS_HPP

#include "grid.hpp"
#include "p4est_dd.hpp"
#include "utils.hpp"

#if (defined(LB_ADAPTIVE) || defined (ES_ADAPTIVE) || defined(EK_ADAPTIVE) || defined(DD_P4EST))

#include <cstdint>
#include <memory>
#include <p4est_to_p8est.h>
#include <p8est.h>
#include <p8est_connectivity.h>
#include <p8est_ghost.h>
#include <p8est_mesh.h>
#include <p8est_meshiter.h>
#include <p8est_virtual.h>
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

/** Allow using std::lower_bound with sc_arrays by stubbing begin and end
 */
template <typename T>
T* sc_wrap_begin(const sc_array_t *a) {
  assert(sizeof(T) == a->elem_size);
  return reinterpret_cast<T*>(a->array);
}

template <typename T>
T* sc_wrap_end(const sc_array_t *a) {
  return sc_wrap_begin<T>(a) + a->elem_count;
}



/* "global variables" */
#if defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
extern castable_unique_ptr<p4est_t> adapt_p4est;
extern castable_unique_ptr<p4est_connectivity_t> adapt_conn;
extern castable_unique_ptr<p4est_ghost_t> adapt_ghost;
extern castable_unique_ptr<p4est_mesh_t> adapt_mesh;
extern castable_unique_ptr<p4est_virtual_t> adapt_virtual;
extern castable_unique_ptr<p4est_virtual_ghost_t> adapt_virtual_ghost;
#ifdef COMM_HIDING
extern std::vector<p8est_virtual_ghost_exchange_t*> exc_status_lb;
#endif
#endif

/** Struct containing information to adjust p4est behavior */
typedef struct {
  /** minimum required grid level */
  int min_ref_level;
  /** maximum allowed grid level */
  int max_ref_level;

  // save meshwidth for all available levels
  double h[P8EST_MAXLEVEL];

  // prefactors based on p4est_params.max_ref_level
  double prefactors[P8EST_MAXLEVEL];

  /** partitioning strategy:
   * allowed values are "n_cells" and "subcycling".
   * n_cells will assign each cell a weight of 1 while subcycling will assign
   * each cell the weight 2**(level - min(level)). */
  std::string partitioning;

  /** Relative threshold values for dynamic adaptivity.  The first value gives a
   * lower bound.  If the current cells value (v < first_value * max_value) the
   * cell will be marked for coarsening.  If the current cells value
   * (v > second_value * max_value) it will be marked for refinement.
   * Default values are 0 and 1, such that all criteria have to be enabled
   * manually by the user. */
#ifdef LB_ADAPTIVE
  /** velocity */
  double threshold_velocity[2];
  /** vorticity */
  double threshold_vorticity[2];
#endif // LB_ADAPTIVE
} p4est_parameters;

extern p4est_parameters p4est_params;

#ifdef LB_ADAPTIVE
extern int lb_conn_brick[3];
#endif // LB_ADAPTIVE

extern double coords_for_regional_refinement[6]; // order: x_min, x_max,
                                                 //        y_min, y_max,
                                                 //        z_min, z_max
extern double vel_reg_ref[3];

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
  std::vector<int64_t> p4est_space_idx;
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

/** Returns true if the lb has been initialized.
 */
bool adaptive_lb_is_active();

/** Returns a const reference to the forest_info of "fo".
 * Throws a std::out_of_range if the forest_info of "fo" has not been created
 * yet.
 *
 * @param fo specifier which forest_info to return
 */
const p4est_utils_forest_info_t &p4est_utils_get_forest_info(forest_order fo);

/** Prepares the use of p4est_utils with the short-range MD tree and
 * possibly also the adapt_LB tree.
 */
void p4est_utils_init();

/**
 * Destroy and rebuild forest_info, partition the forests, and recreate
 * p4est_ghost, p4est_mesh, p4est_virtual, and p4est_virtual_ghost.
 * This function leaves payload untouched and is most helpful for static grid
 * refinement before the simulation has started.
 *
 * @param btype      Neighbor information that is to be included
 * @param partition  Should the p4est be repartitioned while structs are built
 */
void p4est_utils_rebuild_p4est_structs(p4est_connect_type_t btype,
                                       bool partition = true);

/*****************************************************************************/
/** \name Python interface                                                   */
/*****************************************************************************/
/*@{*/
/** Set min level from p4est actor */
#if defined(LB_ADAPTIVE) || defined(EK_ADAPTIVE) || defined(ES_ADAPTIVE)
inline int p4est_utils_set_min_level(int lvl) {
  p4est_params.min_ref_level = lvl;
  mpi_call(mpi_set_min_level, -1, lvl);
  return 0;
}

/** Get min level from p4est actor */
inline int p4est_utils_get_min_level(int *lvl) {
  *lvl = p4est_params.min_ref_level;
  return 0;
}

/** Set max level from p4est actor */
inline int p4est_utils_set_max_level(int lvl) {
  p4est_params.max_ref_level = lvl;
  mpi_call(mpi_set_max_level, -1, lvl);
  return 0;
}

/** Get max level from p4est actor */
inline int p4est_utils_get_max_level(int *lvl) {
  *lvl = p4est_params.max_ref_level;
  return 0;
}

/** Set partitioning strategy from p4est actor */
inline int p4est_utils_set_partitioning(std::string part) {
  if (!strcmp(part.data(), "n_cells") || !strcmp(part.data(), "subcycling")) {
    p4est_params.partitioning.assign(part);
    mpi_call(mpi_set_partitioning, -1, 0);
    mpi_set_partitioning(0, 0);
    return 0;
  }
  else {
    return -1;
  }
}

/** Get partitioning strategy from p4est actor */
inline int p4est_utils_get_partitioning(std::string *part) {
  part->assign(p4est_params.partitioning.data());
  return 0;
}

/** Set velocity threshold values for dynamic grid change */
inline int p4est_utils_set_threshold_velocity(double *thresh) {
#ifdef LB_ADAPTIVE
  memcpy(p4est_params.threshold_velocity, thresh, 2 * sizeof(double));
  mpi_call(mpi_bcast_thresh_vel,-1, -1);
  mpi_bcast_thresh_vel(0, 0);
#endif // LB_ADAPTIVE
  return 0;
}

/** Get velocity threshold values for dynamic grid change */
inline int p4est_utils_get_threshold_velocity(double *thresh) {
#ifdef LB_ADAPTIVE
  for (int i = 0; i < 2; ++i)
    thresh[i] = p4est_params.threshold_velocity[i];
#endif // LB_ADAPTIVE
  return 0;
}

/** Set vorticity threshold values for dynamic grid change */
inline int p4est_utils_set_threshold_vorticity(double *thresh) {
#ifdef LB_ADAPTIVE
  memcpy(p4est_params.threshold_vorticity, thresh, 2 * sizeof(double));
  mpi_call(mpi_bcast_thresh_vort,-1, -1);
  mpi_bcast_thresh_vort(0, 0);
#endif // LB_ADAPTIVE
  return 0;
}
/** Get vorticity threshold values for dynamic grid change */
inline int p4est_utils_get_threshold_vorticity(double *thresh) {
#ifdef LB_ADAPTIVE
  for (int i = 0; i < 2; ++i)
    thresh[i] = p4est_params.threshold_vorticity[i];
#endif // LB_ADAPTIVE
  return 0;
}

/** Perform uniform grid refinement ref_step times */
inline int p4est_utils_uniform_refinement(int ref_steps) {
  mpi_call(mpi_unif_refinement, -1, ref_steps);
  mpi_unif_refinement(0, ref_steps);
  return 0;
}

/** Perform random grid refinement ref_step times */
inline int p4est_utils_random_refinement(int ref_steps) {
  mpi_call(mpi_rand_refinement, -1, ref_steps);
  mpi_rand_refinement(0, ref_steps);
  return 0;
}

/** Perform regional coarsening within given bounding box coordinates */
inline int p4est_utils_regional_coarsening(double *bb_coords) {
  memcpy(coords_for_regional_refinement, bb_coords, 6 * sizeof(double));
  mpi_call(mpi_bcast_parameters_for_regional_refinement, -1, 0);
  mpi_bcast_parameters_for_regional_refinement(0, 0);
  mpi_call(mpi_reg_coarsening, -1, 0);
  mpi_reg_coarsening(0, 0);
  return 0;
}

/** Perform regional refinement within given bounding box coordinates */
inline int p4est_utils_regional_refinement(double *bb_coords) {
  memcpy(coords_for_regional_refinement, bb_coords, 6 * sizeof(double));
  mpi_call(mpi_bcast_parameters_for_regional_refinement, -1, 0);
  mpi_bcast_parameters_for_regional_refinement(0, 0);
  mpi_call(mpi_reg_refinement, -1, 0);
  mpi_reg_refinement(0, 0);
  return 0;
}

/** Exclude a boundary index from geometric refinement or inverse geometric
 * refinement.
 */
inline int p4est_utils_geometric_refinement_exclude_boundary_index(int index) {
  mpi_call(mpi_exclude_boundary, -1, index);
  mpi_exclude_boundary(0, index);
  return 0;
}

/** Perform geometric refinement step, i.e. refine around boundaries. */
inline int p4est_utils_geometric_refinement() {
  mpi_call(mpi_geometric_refinement, -1, 0);
  mpi_geometric_refinement(0, 0);
  return 0;
}

/** Perform inverse geometric refinement step, i.e. refine everywhere but around
 * boundaries.
 */
inline int p4est_utils_inverse_geometric_refinement() {
  mpi_call(mpi_inv_geometric_refinement, -1, 0);
  mpi_inv_geometric_refinement(0, 0);
  return 0;
}

/** Trigger dynamic grid adaptation. */
inline int p4est_utils_adapt_grid() {
  if (p4est_params.min_ref_level < p4est_params.max_ref_level) {
    mpi_call(mpi_adapt_grid, -1, 0);
    mpi_adapt_grid(0, 0);
  }
  return 0;
}

/** Set an area of refinement and prepare initial grid */
inline int p4est_utils_set_refinement_area(double *bb_coords, double *vel) {
  std::memcpy(coords_for_regional_refinement, bb_coords, 6 * sizeof(double));
  std::memcpy(vel_reg_ref, vel, 3 * sizeof(double));
  mpi_call(mpi_set_refinement_area, -1, 0);
  mpi_set_refinement_area(0, 0);
  for (int i = 0; i < (p4est_params.max_ref_level - p4est_params.min_ref_level);
       ++i) {
    mpi_call(mpi_reg_refinement, -1, 0);
    mpi_reg_refinement(0, 0);
  }
  return 0;
}
#endif // defined(LB_ADAPTIVE) || defined(EK_ADAPTIVE) || defined(ES_ADAPTIVE)
/*@}*/

/*****************************************************************************/
/** \name Geometric helper functions                                         */
/*****************************************************************************/
/*@{*/
//--------------------------------------------------------------------------------------------------

// Compute the grid- and bricksize according to box_l and maxrange

/** Determine the level that is needed to discretize ncells with per direction
 * with a given mesh width.
 */
int p4est_utils_determine_grid_level(double mesh_width,
                                     std::array<int, 3> &ncells);

/** Determine the optimal number of trees to discretize the domain with a given
 * mesh_width.
 */
void p4est_utils_set_cellsize_optimal(double mesh_width);

/** Transform coordinates in-place from p4est-tree coordinates (a tree is a
 * unit-cube) to simulation domain.
 *
 * @param x    Position that is transformed in place.
 */
static inline void tree_to_boxlcoords(double x[3]) {
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

/** Transform coordinates from p4est tree-coordinates to simulation domain
 *
 * @param x    Position to be transformed.
 * @return     Position in simulation domain
 */
static inline std::array<double, 3> tree_to_boxlcoords_copy(double x[3]) {
  std::array<double, 3> res = {{x[0], x[1], x[2]}};
  tree_to_boxlcoords(res.data());
  return res;
}

/** Transform coordinates in-place from simulation domain to p4est
 * tree-coordinates (a tree is a unit-cube).
 *
 * @param x    Position that is transformed in place.
 */
static inline void boxl_to_treecoords(double x[3]) {
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

/** Transform coordinates from simulation domain to p4est tree-coordinates
 *
 * @param x    Position to be transformed.
 * @return     Position in p4est tree-coordinates
 */
static inline std::array<double, 3> boxl_to_treecoords_copy(double x[3]) {
  std::array<double, 3> res = {{x[0], x[1], x[2]}};
  boxl_to_treecoords(res.data());
  return res;
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

/** Check if two quadrants touch
 *
 * \param [in]  q1, q2     The two quadrants
 * \param [in] tree1, tree2  Tree id of respective quadrants
 * \return                 True if the quadrants touch via face, edge, or corner
 */
bool p4est_utils_quadrants_touching(p4est_quadrant_t* q1, p4est_topidx_t tree1,
                                    p4est_quadrant_t* q2, p4est_topidx_t tree2);

/** Verify that a quadrant contains the given position
 *
 * @param qid      Quadrant index 0 .. local_num_quadrants for local quadrants
 *                 or 0 .. ghost_num_quadrants for ghost quadrants
 * @param pos      Position that is supposed to be contained within quadrant
 * @param ghost    True if index is a ghost index
 * @return         True if quadrant extends over spatial position
 */
bool p4est_utils_pos_sanity_check(p4est_locidx_t qid, double pos[3],
                                  bool ghost = false);

/** Verify that quadrants are actually touching
 *
 * @param pos_mp_q1, pos_mp_q2       Position of midpoint of quadrants
 * @param level_q1, level_q2   Level of quadrants
 */
bool p4est_utils_pos_vicinity_check(std::array<double, 3> pos_mp_q1,
                                    int level_q1,
                                    std::array<double, 3> pos_mp_q2,
                                    int level_q2);

/** Verify that position is in fact between those two found quadrants
 *
 * @param pos_mp_q1, pos_mp_q2  Midpoints of involved quadrants
 * @param pos                   Position that we are interested in
 */
bool p4est_utils_pos_enclosing_check(const std::array<double, 3> &pos_mp_q1,
                                     const int level_q1,
                                     const std::array<double, 3> &pos_mp_q2,
                                     const int level_q2,
                                     const double pos[3],
                                     std::array<double, 6> &interpol_weights);

#if defined(LB_ADAPTIVE) || defined (EK_ADAPTIVE) || defined (ES_ADAPTIVE)
/** Binary search for a local quadrant within the adaptive p4est */
p4est_locidx_t p4est_utils_bin_search_quad(p4est_gloidx_t index);

/** Binary search for a quadrant within mirrors or ghosts.
 *
 * @param search_index  Quadrant index that we search
 * @param level         Level of quadrant we search
 * @param search_space  Array of quadrants on which we conduct binary search
 * @param result        Initially empty array containing all matching quadrants
 */
void p4est_utils_bin_search_quad_in_array(uint64_t search_index,
                                          sc_array_t * search_space,
                                          std::vector<p4est_locidx_t> &result,
                                          int level = -1);
#endif

/** Split a Morton-index into 3 integers, i.e. the position on a virtual regular
 * grid on the finest level */
std::array<uint64_t, 3> p4est_utils_idx_to_pos (uint64_t idx);

/** Obtain a Morton-index by interleaving 3 integer coordinates.
 *
 * @param x, y, z  The coordinates
 * @return         Morton-index of given position.
 */
int64_t p4est_utils_cell_morton_idx(uint64_t x, uint64_t y, uint64_t z);

/** Obtain a Morton-index from a quadrant and its treeid.
 *
 * @param fi       Forest info containing p4est
 * @param q        Quadrant of a given tree.
 * @param tree     Tree-id holding given quadrant.
 * @param displace Optional offset-argument to move index coordinates
 * @return         Morton-index of given quadrant.
 */
int64_t p4est_utils_global_idx(p4est_utils_forest_info_t fi,
                               const p8est_quadrant_t *q,
                               const p4est_topidx_t tree,
                               const std::array<int, 3> displace = {{0, 0, 0}});

/** Calculate a Morton-index from a position in Cartesian coordinates
 *
 * @param forest   The position of the forest info containing the p4est for
 *                 which to map.
 * @param xyz      The position to map.
 * @return         Morton index of position in given p4est.
 */
int64_t p4est_utils_pos_to_index(forest_order forest, double xyz[3]);

/** Map a geometric position to the respective processor.
 *
 * @param forest   The position of the forest info containing the p4est for
 *                 which to map.
 * @param xyz      The position to map.
 * @return         The MPI rank which holds a quadrant at the given position for
 *                 the current p4est.
 */
p4est_locidx_t p4est_utils_pos_to_qid(forest_order forest, double *xyz);

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
static int p4est_utils_allocate_levelwise_storage(
    std::vector<std::vector<T>> &data, p4est_mesh_t *mesh,
    p4est_virtual_t *virtual_quads, bool local_data) {
  if (!data.empty()) {
    p4est_utils_deallocate_levelwise_storage(data);
  }

  // allocate data for each level
  data = std::vector<std::vector<T>>(P8EST_QMAXLEVEL, std::vector<T>());
  P4EST_ASSERT(data.size() == P8EST_QMAXLEVEL);

  size_t quads_on_level;

  for (size_t level = 0; level < P8EST_QMAXLEVEL; ++level) {
    quads_on_level =
        local_data
            ? (mesh->quad_level + level)->elem_count +
                  P8EST_CHILDREN * (virtual_quads->virtual_qlevels +
                                    level)->elem_count
            : (mesh->ghost_level + level)->elem_count +
                  P8EST_CHILDREN * (virtual_quads->virtual_glevels +
                                    level)->elem_count;
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
static int p4est_utils_deallocate_levelwise_storage(
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
static int prepare_ghost_exchange(std::vector<std::vector<T>> &local_data,
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
#if defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
#ifdef COMM_HIDING
int p4est_utils_end_pending_communication(
    std::vector<p8est_virtual_ghost_exchange_t*> &exc_status, int level = -1);
#endif

/** Function that handles grid alteration. After calling this function the grid
 * has changed and everything is set to perform the next integration step.
 */
int p4est_utils_perform_adaptivity_step();
#endif

#if defined(DD_P4EST) || defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
template <typename T>
/** Flatten level-wise data, i.e. inverse operation of
 * \ref p4est_utils_unflatten_data
 *
 * @tparam T                 Data type of numerical payload.
 * @param p4est              The forest whose numerical payload is to be
 *                           flattened.
 * @param mesh               Mesh structure of current p4est.
 * @param virtual_quads      Virtual quadrants containing the mapping between
 *                           quadrant id and its storage id.
 * @param data_levelwise     Level-wise data-structure containing original data.
 * @param data_flat          Flat data-structure that will be populated.
 */
int p4est_utils_flatten_data(p8est_t *p4est, p8est_mesh_t *mesh,
                             p8est_virtual_t *virtual_quads,
                             std::vector< std::vector<T> > &data_levelwise,
                             std::vector<T> &data_flat) {
  p8est_quadrant_t *q;
  data_flat.resize(p4est->local_num_quadrants);
  for (p4est_locidx_t qid = 0; qid < p4est->local_num_quadrants; ++qid) {
    q = p4est_mesh_get_quadrant(p4est, mesh, qid);
    std::memcpy(data_flat.data() + qid,
                data_levelwise[q->level].data() +
                virtual_quads->quad_qreal_offset[qid],
                sizeof(T));
  }
  return 0;
}

template <typename T>
/** Unflatten flat data into level-wise datastructure, i.e. inverse operation of
 * \ref p4est_utils_flatten_data
 *
 * @tparam T                 Data type of numerical payload.
 * @param p4est              The forest whose numerical payload is to be
 *                           unflattened.
 * @param mesh               Mesh structure of current p4est.
 * @param virtual_quads      Virtual quadrants containing the mapping between
 *                           quadrant id and its storage id.
 * @param data_flat          Flat data-structure containing original data
 * @param data_levelwise     Level-wise data-structure to be populated
 */
int p4est_utils_unflatten_data(p8est_t *p4est, p8est_mesh_t *mesh,
                               p8est_virtual_t *virtual_quads,
                               std::vector<T> &data_flat,
                               std::vector< std::vector<T> > &data_levelwise) {
  p8est_quadrant_t *q;
  for (p4est_locidx_t qid = 0; qid < p4est->local_num_quadrants; ++qid) {
    q = p4est_mesh_get_quadrant(p4est, mesh, qid);
    std::memcpy(data_levelwise[q->level].data() +
                virtual_quads->quad_qreal_offset[qid],
                data_flat.data() + qid,
                sizeof(T));
  }
  return 0;
}
#endif

#if defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
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
#endif // defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
/*@}*/

/*****************************************************************************/
/** \name Partition adaptive p4ests; Helper functions                        */
/*****************************************************************************/
/*@{*/
#if defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
/** Get weights based on given metric
 *
 * @param metric   String of metric. Currently supported are
 *                 n_cells      Each cell is weighted as 1.
 *                 subcycling   Each cell is weighted with 2**(l_max - l_curr)
 * @return
 */
std::vector<double> p4est_utils_get_adapt_weights(const std::string& metric);

/** Preprocess repartitioning, i.e. store former partitioning table.
 */
int p4est_utils_repart_preprocess();

/** Postprocess repartitioning, i.e. move data to respective processors.
 */
int p4est_utils_repart_postprocess();
#endif // defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
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
