#include "p4est_utils.hpp"

#if (defined(LB_ADAPTIVE) || defined(DD_P4EST))

#include "debug.hpp"
#include "domain_decomposition.hpp"
#include "grid.hpp"
#include "lb-adaptive.hpp"
#include "lbmd_repart.hpp"
#include "p4est_dd.hpp"
#include "utils/Morton.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>

#include <mpi.h>

#include <p8est_algorithms.h>
#include <p8est_bits.h>
#include <p8est_communication.h>
#include <p8est_search.h>

static std::vector<p4est_utils_forest_info_t> forest_info;

#if defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
castable_unique_ptr<p4est_t> adapt_p4est;
castable_unique_ptr<p4est_connectivity_t> adapt_conn;
castable_unique_ptr<p4est_ghost_t> adapt_ghost;
castable_unique_ptr<p4est_mesh_t> adapt_mesh;
castable_unique_ptr<p4est_virtual_t> adapt_virtual;
castable_unique_ptr<p4est_virtual_ghost_t> adapt_virtual_ghost;
#ifdef COMM_HIDING
std::vector<p8est_virtual_ghost_exchange_t*> exc_status_lb (19, nullptr);
#endif
#endif

p4est_parameters p4est_params = {
  // min_ref_level
  -1,
  // max_ref_level
  -1,
 // h
  {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
  // prefactors
  {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
  // partitioning
  "n_cells",
#ifdef LB_ADAPTIVE
  // threshold_velocity
  {0.0, 1.0},
  // threshold_vorticity
  {0.0, 1.0},
#endif
};

#ifdef LB_ADAPTIVE
int lb_conn_brick[3] = {0, 0, 0};
#endif // LB_ADAPTIVE

std::array<double, 6> coords_for_regional_refinement =
    {{std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
      std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
      std::numeric_limits<double>::min(), std::numeric_limits<double>::max()}};
double vel_reg_ref[3] = { std::numeric_limits<double>::min(),
                          std::numeric_limits<double>::min(),
                          std::numeric_limits<double>::min() };

// CAUTION: Do ONLY use this pointer in p4est_utils_perform_adaptivity_step
#if defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
std::vector<int> flags;
std::vector<p4est_gloidx_t> old_partition_table_adapt;
#endif // defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)

#ifdef LB_ADAPTIVE
std::vector<lbadapt_payload_t> linear_payload_lbm;
#endif // LB_ADAPTIVE

const p4est_utils_forest_info_t &p4est_utils_get_forest_info(forest_order fo) {
  // Use at() here because forest_info might not have been initialized yet.
  return forest_info.at(static_cast<int>(fo));
}

static p4est_utils_forest_info_t p4est_to_forest_info(p4est_t *p4est) {
  if (p4est) {
    // fill element to insert
    p4est_utils_forest_info_t insert_elem(p4est);

    // only inspect local trees if current process hosts quadrants
    if (p4est->local_num_quadrants != 0) {
      for (p4est_topidx_t i = p4est->first_local_tree;
           i <= p4est->last_local_tree; ++i) {
        p8est_tree_t *tree = p8est_tree_array_index(p4est->trees, i);
        // local max level
        if (insert_elem.finest_level_local < tree->maxlevel) {
          insert_elem.finest_level_local = insert_elem.coarsest_level_local =
              tree->maxlevel;
        }
        // local min level
        for (int l = insert_elem.coarsest_level_local; l >= 0; --l) {
          if (l < insert_elem.coarsest_level_local &&
              tree->quadrants_per_level[l]) {
            insert_elem.coarsest_level_local = l;
          }
        }
      }
    }
    // synchronize level and insert into forest_info vector
    MPI_Allreduce(&insert_elem.finest_level_local,
                  &insert_elem.finest_level_global, 1, P4EST_MPI_LOCIDX, MPI_MAX,
                  p4est->mpicomm);
    MPI_Allreduce(&insert_elem.coarsest_level_local,
                  &insert_elem.coarsest_level_global, 1, P4EST_MPI_LOCIDX,
                  MPI_MIN, p4est->mpicomm);
    insert_elem.finest_level_ghost = insert_elem.finest_level_global;
    insert_elem.coarsest_level_ghost = insert_elem.coarsest_level_global;

    // get Morton-IDs analog to p4est_utils_cell_morton_idx
    insert_elem.p4est_space_idx.resize(n_nodes + 1);
    for (int i = 0; i <= n_nodes; ++i) {
      p4est_quadrant_t *q = &p4est->global_first_position[i];
      if (i < n_nodes && q->p.which_tree < p4est->trees->elem_count) {
        double xyz[3];
        insert_elem.p4est_space_idx[i] =
            p4est_utils_global_idx(insert_elem, q, q->p.which_tree);
      } else {
        double n_trees[3];
        double n_trees_alt[3];
#if defined(LB_ADAPTIVE) && defined(DD_P4EST)
        for (int i = 0; i < P8EST_DIM; ++i) {
          n_trees[i] = lb_conn_brick[i];
          n_trees_alt[i] = dd_p4est_get_n_trees(i);
          P4EST_ASSERT((n_trees[i] == n_trees_alt[i]) ||
                       ((n_trees[i] < n_trees_alt[i]) && n_trees[i] == 0) ||
                       ((n_trees_alt[i] < n_trees[i]) && n_trees_alt[i] == 0));
          if (n_trees[i] < n_trees_alt[i]) {
            P4EST_ASSERT (n_trees[i] == 0);
            n_trees[i] = n_trees_alt[i];
          }
        }
#elif defined(LB_ADAPTIVE)
        for (int i = 0; i < P8EST_DIM; ++i) {
          n_trees[i] = lb_conn_brick[i];
        }
#elif defined(DD_P4EST)
        for (int i = 0; i < P8EST_DIM; ++i)
          n_trees[i] = dd_p4est_get_n_trees(i);
#endif
        int64_t tmp = 1 << insert_elem.finest_level_global;
        while (tmp < ((box_l[0] / n_trees[0]) * (1 << insert_elem.finest_level_global)))
          tmp <<= 1;
        while (tmp < ((box_l[1] / n_trees[1]) * (1 << insert_elem.finest_level_global)))
          tmp <<= 1;
        while (tmp < ((box_l[2] / n_trees[2]) * (1 << insert_elem.finest_level_global)))
          tmp <<= 1;
        insert_elem.p4est_space_idx[i] = tmp * tmp * tmp;
      }
    }
    return insert_elem;
  }
  else {
    return p4est_utils_forest_info_t(nullptr);
  }
}

/** For algorithms like mapping a position to a quadrant to work we need a
 * synchronized version of the quadrant offsets of each tree.
 *
 * @param p4ests     List of all p4ests in the current simulation
 */
void p4est_utils_prepare(std::vector<p8est_t *> p4ests) {
  forest_info.clear();

  std::transform(std::begin(p4ests), std::end(p4ests),
                 std::back_inserter(forest_info), p4est_to_forest_info);
}

void p4est_utils_init() {
  p4est_utils_prepare({
    dd_p4est_get_p4est(),
#ifdef LB_ADAPTIVE
    adapt_p4est,
#endif
  });
}

void p4est_utils_rebuild_p4est_structs(p4est_connect_type_t btype,
                                       bool partition) {
  std::vector<p4est_t *> forests;
#ifdef DD_P4EST
  forests.push_back(dd_p4est_get_p4est());
#endif // DD_P4EST
#ifdef LB_ADAPTIVE
  forests.push_back(adapt_p4est);
#endif // LB_ADAPTIVE
  p4est_utils_prepare(forests);

  if (partition) {
#if defined(DD_P4EST) && defined(LB_ADAPTIVE)
    // create aligned trees
    auto adapt_forest_info =
        forest_info.at(static_cast<size_t>(forest_order::adaptive_LB));
    if (adapt_forest_info.coarsest_level_global ==
        adapt_forest_info.finest_level_global) {
      auto short_range_forest_info = forest_info.at(
          static_cast<size_t>(forest_order::short_range));
      auto mod = (short_range_forest_info.coarsest_level_global <=
                  adapt_forest_info.coarsest_level_global) ?
                 forest_order::adaptive_LB : forest_order::short_range;
      auto ref = (short_range_forest_info.coarsest_level_global <=
                  adapt_forest_info.coarsest_level_global) ?
                 forest_order::short_range : forest_order::adaptive_LB;
      p4est_dd_repart_preprocessing();
      p4est_utils_partition_multiple_forests(ref, mod);
      p4est_utils_prepare(forests);

      // copied from lbmd_repart short-range MD postprocess
      cells_re_init(CELL_STRUCTURE_CURRENT, true, true);
    }

    // do the partitioning
    p4est_dd_repart_preprocessing();
    std::vector<std::string> metrics;
    std::vector<double> alpha = {1., 1.};
    std::vector<double> weights_md(dd_p4est_get_p4est()->local_num_quadrants,
                                   1.0);
    std::vector<double> weights_lb =
        p4est_utils_get_adapt_weights(p4est_params.partitioning);
    p4est_dd_repart_preprocessing();
    p4est_utils_weighted_partition(dd_p4est_get_p4est(), weights_md, 1.0,
                                   adapt_p4est, weights_lb, 1.0);
    cells_re_init(CELL_STRUCTURE_CURRENT, true, true);
    p4est_utils_prepare(forests);
#elif defined(LB_ADAPTIVE)
    if (p4est_params.partitioning == "n_cells") {
      p8est_partition_ext(p4est_partitioned, 1,
                          lbadapt_partition_weight_uniform);
    }
    else if (p4est_params.partitioning == "subcycling") {
      p8est_partition_ext(p4est_partitioned, 1,
                          lbadapt_partition_weight_subcycling);
    }
    else {
      SC_ABORT_NOT_REACHED();
    }
#elif defined(DD_P4EST)
    p4est_dd_repart_preprocessing();
    p8est_partition(dd_p4est_get_p4est(), 1, nullptr);
    cells_re_init(CELL_STRUCTURE_CURRENT, true, true);
#endif // DD_P4EST
    p4est_utils_prepare(forests);
  }
#ifdef LB_ADAPTIVE_GPU
  local_num_quadrants = adapt_p4est->local_num_quadrants;
#endif // LB_ADAPTIVE_GPU

#if defined(LB_ADAPTIVE)
  adapt_ghost.reset(p4est_ghost_new(adapt_p4est, btype));
  adapt_mesh.reset(p4est_mesh_new_ext(adapt_p4est, adapt_ghost, 1, 1, 1,
                                      btype));
  adapt_virtual.reset(p4est_virtual_new_ext(adapt_p4est, adapt_ghost,
                                            adapt_mesh, btype, 1));
  adapt_virtual_ghost.reset(p4est_virtual_ghost_new(adapt_p4est, adapt_ghost,
                                                    adapt_mesh, adapt_virtual,
                                                    btype));
#endif // defined(LB_ADAPTIVE)
}

static inline int count_trailing_zeros(int x)
{
  int z = 0;
  for (; (x & 1) == 0; x >>= 1) z++;
  return z;
}

int p4est_utils_determine_grid_level(double mesh_width,
                                     std::array<int, 3> &ncells) {
  // compute number of cells
  if (mesh_width > ROUND_ERROR_PREC * box_l[0]) {
    ncells[0] = std::max<int>(box_l[0] / mesh_width, 1);
    ncells[1] = std::max<int>(box_l[1] / mesh_width, 1);
    ncells[2] = std::max<int>(box_l[2] / mesh_width, 1);
  }

  // divide all dimensions by biggest common power of 2
  return count_trailing_zeros(ncells[0] | ncells[1] | ncells[2]);
}

void p4est_utils_set_cellsize_optimal(double mesh_width) {
#ifdef LB_ADAPTIVE
  std::array<int, 3> ncells = {{1, 1, 1}};

  int grid_level = p4est_utils_determine_grid_level(mesh_width, ncells);

  lb_conn_brick[0] = ncells[0] >> grid_level;
  lb_conn_brick[1] = ncells[1] >> grid_level;
  lb_conn_brick[2] = ncells[2] >> grid_level;
#endif
}

void p4est_utils_get_front_lower_left(p8est_t *p8est, p4est_topidx_t which_tree,
                                      p8est_quadrant_t *q, double *xyz) {
  p8est_qcoord_to_vertex(p8est->connectivity, which_tree, q->x, q->y, q->z,
                         xyz);
  tree_to_boxlcoords(xyz);
}

void p4est_utils_get_front_lower_left(p8est_meshiter_t *mesh_iter,
                                      double *xyz) {
  p8est_quadrant_t *q = p8est_mesh_get_quadrant(
      mesh_iter->p4est, mesh_iter->mesh, mesh_iter->current_qid);
  p8est_qcoord_to_vertex(mesh_iter->p4est->connectivity,
                         mesh_iter->mesh->quad_to_tree[mesh_iter->current_qid],
                         q->x, q->y, q->z, xyz);
  tree_to_boxlcoords(xyz);
}

void p4est_utils_get_midpoint(p8est_t *p8est, p4est_topidx_t which_tree,
                              p8est_quadrant_t *q, double xyz[3]) {
  int base = P8EST_QUADRANT_LEN(q->level);
  int root = P8EST_ROOT_LEN;
  double half_length = ((double)base / (double)root) * 0.5;
  p8est_qcoord_to_vertex(p8est->connectivity, which_tree, q->x, q->y, q->z,
                         xyz);
  for (int i = 0; i < P8EST_DIM; ++i) {
    xyz[i] += half_length;
  }
  tree_to_boxlcoords(xyz);
}

void p4est_utils_get_midpoint(p8est_meshiter_t *mesh_iter, double *xyz) {
  int base = P8EST_QUADRANT_LEN(mesh_iter->current_level);
  int root = P8EST_ROOT_LEN;
  double half_length = ((double)base / (double)root) * 0.5;

  p8est_quadrant_t *q = p8est_mesh_get_quadrant(
      mesh_iter->p4est, mesh_iter->mesh, mesh_iter->current_qid);
  p8est_qcoord_to_vertex(mesh_iter->p4est->connectivity,
                         mesh_iter->mesh->quad_to_tree[mesh_iter->current_qid],
                         q->x, q->y, q->z, xyz);

  for (int i = 0; i < P8EST_DIM; ++i) {
    xyz[i] += half_length;
  }
  tree_to_boxlcoords(xyz);
}


#if defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
bool p4est_utils_quadrants_touching(p4est_quadrant_t* q1, p4est_topidx_t tree1,
                                    p4est_quadrant_t* q2, p4est_topidx_t tree2)
{
  std::array<double, 3> pos1, pos2, p1, p2;
  double h1 = p4est_params.h[q1->level];
  double h2 = p4est_params.h[q2->level];
  p4est_utils_get_front_lower_left(adapt_p4est, tree1, q1, pos1.data());
  p4est_utils_get_front_lower_left(adapt_p4est, tree2, q2, pos2.data());

  bool touching = false;
  int fold[3] = {0, 0, 0};
  for (int i = 0; !touching && i < P8EST_CHILDREN; ++i) {
    p1 = pos1;
    if (i & 1) {
      p1[0] += h1;
    }
    if (i & 2) {
      p1[1] += h1;
    }
    if (i & 4) {
      p1[2] += h1;
    }
    fold_position(p1, fold);

    for (int j = 0; j < P8EST_CHILDREN; ++j) {
      p2 = pos2;
      if (j & 1) {
        p2[0] += h2;
      }
      if (j & 2) {
        p2[1] += h2;
      }
      if (j & 4) {
        p2[2] += h2;
      }
      fold_position(p2, fold);
      touching = (p1 == p2);
      if (touching) break;
    }
  }

  return touching;
}

bool p4est_utils_pos_sanity_check(p4est_locidx_t qid, double pos[3], bool ghost) {
  std::array<double, 3> qpos;
  p8est_quadrant_t *quad;
  if (ghost) {
    quad = p4est_quadrant_array_index(&adapt_ghost->ghosts, qid);
    p4est_utils_get_front_lower_left(adapt_p4est,
                                     adapt_mesh->ghost_to_tree[qid], quad,
                                     qpos.data());
  }
  else {
    quad = p8est_mesh_get_quadrant(adapt_p4est, adapt_mesh, qid);
    p4est_utils_get_front_lower_left(adapt_p4est, adapt_mesh->quad_to_tree[qid],
                                     quad, qpos.data());
  }
  return (
      qpos[0] <= pos[0] + ROUND_ERROR_PREC &&
      pos[0] < qpos[0] + p4est_params.h[quad->level] + ROUND_ERROR_PREC &&
      qpos[1] <= pos[1] + ROUND_ERROR_PREC &&
      pos[1] < qpos[1] + p4est_params.h[quad->level] + ROUND_ERROR_PREC &&
      qpos[2] <= pos[2] + ROUND_ERROR_PREC &&
      pos[2] < qpos[2] + p4est_params.h[quad->level] + ROUND_ERROR_PREC);
}

bool p4est_utils_pos_vicinity_check(std::array<double, 3> pos_mp_q1,
                                    int level_q1,
                                    std::array<double, 3> pos_mp_q2,
                                    int level_q2) {
  bool touching = true;
  for (int i = 0; i < 3; ++i) {
    if (level_q1 == level_q2) {
      touching &=
          (((std::abs(pos_mp_q1[i] - pos_mp_q2[i]) ==
             p4est_params.h[level_q1]) ||
            (box_l[i] - std::abs(pos_mp_q1[i] - pos_mp_q2[i]) ==
             p4est_params.h[level_q1])) ||
           ((std::abs(pos_mp_q1[i] - pos_mp_q2[i]) == 0 ||
            (box_l[i] - std::abs(pos_mp_q1[i] - pos_mp_q2[i]) == 0))));
    } else {
      touching &=
          (((std::abs(pos_mp_q1[i] - pos_mp_q2[i]) ==
             0.5 * (p4est_params.h[level_q1] +
                    p4est_params.h[level_q2])) ||
            (box_l[i] - std::abs(pos_mp_q1[i] - pos_mp_q2[i]) ==
             0.5 * (p4est_params.h[level_q1] +
                    p4est_params.h[level_q2]))) ||
           ((std::abs(pos_mp_q1[i] - pos_mp_q2[i]) ==
             0.5 * p4est_params.h[std::max(level_q1, level_q2)] ||
            (box_l[i] - std::abs(pos_mp_q1[i] - pos_mp_q2[i]) ==
             0.5 * p4est_params.h[std::max(level_q1, level_q2)]))));
    }
  }
  return touching;
}

bool p4est_utils_pos_enclosing_check(const std::array<double, 3> &pos_mp_q1,
                                     const int level_q1,
                                     const std::array<double, 3> &pos_mp_q2,
                                     const int level_q2,
                                     const std::array<double, 3> pos,
                                     std::array<double, 6> &interpol_weights) {
  double smaller, larger;
  bool inbetween = true;
  if (level_q1 == level_q2) {
    for (int i = 0; i < 3; ++i) {
      smaller = (pos_mp_q1[i] <= pos_mp_q2[i]) ? pos_mp_q1[i] : pos_mp_q2[i];
      larger = (smaller == pos_mp_q2[i]) ? pos_mp_q1[i] : pos_mp_q2[i];
      if (pos[i] <= 0.5 * p4est_params.h[level_q1]) {
        double temp = larger;
        larger = smaller;
        smaller = temp - box_l[i];
      } else if (box_l[i] - pos[i] < 0.5 * p4est_params.h[level_q1]) {
        double temp = smaller;
        smaller = larger;
        larger = box_l[i] + temp;
      }
      if (smaller != larger) {
        inbetween &= (smaller <= pos[i] && pos[i] <= larger);
      }
      interpol_weights[i] = (pos[i] - smaller) / (larger - smaller);
      interpol_weights[i + 3] = 1.0 - interpol_weights[i];
    }
  } else {

  }
  return inbetween;
}
#endif // defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)

std::array<uint64_t, 3> p4est_utils_idx_to_pos(uint64_t idx) {
  return Utils::morton_idx_to_coords(idx);
}

// Returns the morton index for given cartesian coordinates.
// Note: This is not the index of the p4est quadrants. But the ordering is the same.
int64_t p4est_utils_cell_morton_idx(uint64_t x, uint64_t y, uint64_t z) {
  return Utils::morton_coords_to_idx(x, y, z);
}

int64_t p4est_utils_global_idx(p4est_utils_forest_info_t fi,
                               const p8est_quadrant_t *q,
                               const p4est_topidx_t which_tree,
                               const std::array<int, 3> displace) {
  int x, y, z;
  double xyz[3];
  p8est_qcoord_to_vertex(fi.p4est->connectivity, which_tree, q->x, q->y, q->z, xyz);
  x = xyz[0] * (1 << fi.finest_level_global) + displace[0];
  y = xyz[1] * (1 << fi.finest_level_global) + displace[1];
  z = xyz[2] * (1 << fi.finest_level_global) + displace[2];

  double n_trees[3];
  double n_trees_alt[3];
#if defined(LB_ADAPTIVE) && defined(DD_P4EST)
  for (int i = 0; i < P8EST_DIM; ++i) {
    n_trees[i] = lb_conn_brick[i];
    n_trees_alt[i] = dd_p4est_get_n_trees(i);
    P4EST_ASSERT((n_trees[i] == n_trees_alt[i]) ||
                 ((n_trees[i] < n_trees_alt[i]) && n_trees[i] == 0) ||
                 ((n_trees_alt[i] < n_trees[i]) && n_trees_alt[i] == 0));
    if (n_trees[i] < n_trees_alt[i]) {
      P4EST_ASSERT (n_trees[i] == 0);
      n_trees[i] = n_trees_alt[i];
    }
  }
#elif defined(LB_ADAPTIVE)
  for (int i = 0; i < P8EST_DIM; ++i) {
          n_trees[i] = lb_conn_brick[i];
        }
#elif defined(DD_P4EST)
        for (int i = 0; i < P8EST_DIM; ++i)
          n_trees[i] = dd_p4est_get_n_trees(i);
#endif

  // fold
  int ub = n_trees[0] * (1 << fi.finest_level_global);
  if (x >= ub) x -= ub;
  if (x < 0) x += ub;
  ub = n_trees[1] * (1 << fi.finest_level_global);
  if (y >= ub) y -= ub;
  if (y < 0) y += ub;
  ub = n_trees[2] * (1 << fi.finest_level_global);
  if (z >= ub) z -= ub;
  if (z < 0) z += ub;

  return p4est_utils_cell_morton_idx(x, y, z);
}


int64_t p4est_utils_global_idx(forest_order forest, const p8est_quadrant_t *q,
                               p4est_topidx_t tree) {
  const auto &fi = forest_info.at(static_cast<int>(forest));
  return p4est_utils_global_idx(fi, q, tree);
}

int64_t p4est_utils_pos_to_index(forest_order forest, const double xyz[3]) {
  const auto &fi = forest_info.at(static_cast<int>(forest));
  std::array<double, 3> pos = boxl_to_treecoords_copy(xyz);
  int xyz_mod[3];
  for (int i = 0; i < P8EST_DIM; ++i)
    xyz_mod[i] = pos[i] * (1 << fi.finest_level_global);
  return p4est_utils_cell_morton_idx(xyz_mod[0], xyz_mod[1], xyz_mod[2]);
}

#if defined(LB_ADAPTIVE) || defined (EK_ADAPTIVE) || defined (ES_ADAPTIVE)
/** Perform a binary search for a given quadrant.
 * CAUTION: This only makes sense for adaptive grids and therefore we implicitly
 *          assume that we are operating on the potentially adaptive p4est of
 *          the LBM.
 *
 * @param idx      The index calculated by \ref p4est_utils_cell_morton_idx
 *                 that we are looking for.
 * @return         The qid of this quadrant in the adaptive p4est.
 */
p4est_locidx_t bin_search_loc_quads(p4est_gloidx_t idx) {
  p4est_gloidx_t cmp_idx = -1;
  p4est_locidx_t count, step, index, first;
  p4est_quadrant_t *q;
  p4est_locidx_t zlvlfill = 0;
  first = 0;
  count = adapt_p4est->local_num_quadrants;
  while (0 < count) {
    step = 0.5 * count;
    index = first + step;
    q = p4est_mesh_get_quadrant(adapt_p4est, adapt_mesh, index);
    p4est_topidx_t tid = adapt_mesh->quad_to_tree[index];
    cmp_idx = p4est_utils_global_idx(forest_order::adaptive_LB, q, tid);
    zlvlfill = 1 << (3*(p4est_params.max_ref_level - q->level));
    if (cmp_idx <= idx && idx < cmp_idx + zlvlfill) {
      return index;
    } else if (cmp_idx < idx) {
      // if we found something smaller: move to latter part of search space
      first = index + 1;
      count -= step + 1;
    } else {
      // else limit search space to half the array
      count = step;
    }
  }
  if (cmp_idx <= idx && idx < cmp_idx + zlvlfill) {
    return first;
  } else {
    return -1;
  }
}

p4est_locidx_t p4est_utils_bin_search_quad(p4est_gloidx_t index) {
  return bin_search_loc_quads(index);
}

void p4est_utils_bin_search_quad_in_array(uint64_t search_index,
                                          sc_array_t * search_space,
                                          std::vector<p4est_locidx_t> &result,
                                          int level) {
  const auto fo = forest_order::adaptive_LB;
  const auto begin = sc_wrap_begin<p8est_quadrant_t>(search_space);
  const auto end = sc_wrap_end<p8est_quadrant_t>(search_space);
  auto p = std::upper_bound(
      begin, end, search_index,
      [&](const uint64_t index, const p8est_quadrant_t a) {
        return (index < p4est_utils_global_idx(fo, &a, a.p.which_tree));
      });
  // no level is specified if we search a quadrant containing a specific
  // position
  if (level == -1) {
    if (p != end) {
      p = (p == begin ? p : p - 1);
      result.push_back(std::distance(begin, p));
    }
  } else {
    if (p != end) {
      p = (p == begin ? p : p - 1);
      // here we search for neighboring quadrants that fall within the region
      // spanned by the shifted anchor point and the quadrant size.
      // From 2:1 balancing there are three different cases.
      // 2 of which are trivial: A same size quadrant and a double sized quadrant
      // will always be the quadrant before p.
      // smaller quadrants will need to be copied from p - 1 to the upper bound q
      // which is the first quadrant larger than search_index + zarea
      int zsize = 1 << (p4est_params.max_ref_level - level);
      int zarea = zsize * zsize * zsize;
      p8est_quadrant_t *quad = p;
      auto idx = p4est_utils_global_idx(fo, quad, quad->p.which_tree);
      if (search_index <= idx && idx < (search_index + zarea)) {
        if (quad->level == level + 1) {
          auto q = std::upper_bound(
              begin, end, search_index + zarea,
              [&](const uint64_t index, const p8est_quadrant_t a) {
                  return (index <
                          p4est_utils_global_idx(fo, &a, a.p.which_tree));
              });
          if (q != end) {
            Utils::iota_n(std::back_inserter(result), std::distance(p, q - 1),
                          std::distance(begin, p));
          }
        } else {
          result.push_back(std::distance(begin, p));
        }
      }
    }
  }
}
#endif // defined(LB_ADAPTIVE) || defined (EK_ADAPTIVE) || defined (ES_ADAPTIVE)

p4est_locidx_t p4est_utils_pos_to_qid(forest_order forest, const double xyz[3]) {
  return p4est_utils_idx_to_qid(forest, p4est_utils_pos_to_index(forest, xyz));
}

p4est_locidx_t p4est_utils_idx_to_qid(forest_order forest, const p4est_gloidx_t idx) {
#ifdef LB_ADAPTIVE
  P4EST_ASSERT(forest == forest_order::adaptive_LB);
  return bin_search_loc_quads(idx);
#endif
}

// Find the process that handles the position
int p4est_utils_pos_to_proc(forest_order forest, const double xyz[3]) {
  return p4est_utils_idx_to_proc(forest, p4est_utils_pos_to_index(forest, xyz));
}

int p4est_utils_idx_to_proc(forest_order forest, const p4est_gloidx_t idx) {
  const auto &fi = forest_info.at(static_cast<int>(forest));
  auto it = std::upper_bound(
      std::begin(fi.p4est_space_idx), std::end(fi.p4est_space_idx) - 1, idx,
      [](int i, int64_t index) { return i < index; });

  return std::distance(std::begin(fi.p4est_space_idx), it) - 1;
}

// CAUTION: Currently LB only
int coarsening_criteria(p8est_t *p8est, p4est_topidx_t which_tree,
                        p8est_quadrant_t **quads) {
#ifdef LB_ADAPTIVE
  // get quad id
  int qid = quads[0]->p.user_long;
  // do not coarsen newly generated quadrants
  if (qid == -1) return 0;
  // avoid coarser cells than min_refinement_level
  if (quads[0]->level == p4est_params.min_ref_level) return 0;

  int coarsen = 1;
  for (int i = 0; i < P8EST_CHILDREN; ++i) {
    // do not coarsen quadrants at the boundary
    coarsen &= !refine_geometric(p8est, which_tree, quads[i]) &&
               (flags[qid + i] == 2);
  }
  return coarsen;
#else // LB_ADAPTIVE
  return 0;
#endif // LB_ADAPTIVE
}


int refinement_criteria(p8est_t *p8est, p4est_topidx_t which_tree,
                        p8est_quadrant_t *q) {
#ifdef LB_ADAPTIVE
  // get quad id
  int qid = q->p.user_long;

  // perform geometric refinement
  int refine = refine_geometric(p8est, which_tree, q);

  // refine if we have cells marked for refinement
  if ((q->level < p4est_params.max_ref_level) &&
      ((1 == flags[qid] || refine))) {
    return 1;
  }
#endif // LB_ADAPTIVE
  return 0;
}

// use around particles
typedef struct {
  std::array<double, 3> bbox_min;
  std::array<double, 3> bbox_max;
} refinement_area_t;

void create_refinement_patch_from_pos(std::vector<refinement_area_t> &patches,
                                      Vector3d position, double radius) {
  refinement_area_t ref_area;
  std::array<double, 3> img_box = {{0, 0, 0}};
  ref_area.bbox_min = {{position[0] - radius, position[1] - radius,
                        position[2] - radius}};
  fold_position(img_box, ref_area.bbox_min);
  ref_area.bbox_max = {{position[0] + radius, position[1] + radius,
                        position[2] + radius}};
  fold_position(img_box, ref_area.bbox_min);
  patches.push_back(ref_area);
}

void p4est_utils_collect_flags(std::vector<int> &flags) {
#ifdef LB_ADAPTIVE
  // get refinement string for grid change
  // velocity
  double v;
  double v_min = std::numeric_limits<double>::max();
  double v_max = std::numeric_limits<double>::min();
  castable_unique_ptr<sc_array_t> vel_values;
  if (p4est_params.threshold_velocity[0] != 0.0 &&
      p4est_params.threshold_velocity[1] != 1.0) {
    // Euclidean norm
    vel_values =
        sc_array_new_size(sizeof(double), 3 * adapt_p4est->local_num_quadrants);
    lbadapt_get_velocity_values(vel_values);
    for (int qid = 0; qid < adapt_p4est->local_num_quadrants; ++qid) {
      v = sqrt(Utils::sqr(*(double *) sc_array_index(vel_values, 3 * qid)) +
               Utils::sqr(*(double *) sc_array_index(vel_values, 3 * qid + 1)) +
               Utils::sqr(*(double *) sc_array_index(vel_values, 3 * qid + 2)));
      if (v < v_min) {
        v_min = v;
      }
      if (v > v_max) {
        v_max = v;
      }
    }
    // sync
    v = v_min;
    MPI_Allreduce(&v, &v_min, 1, MPI_DOUBLE, MPI_MIN, adapt_p4est->mpicomm);
    v = v_max;
    MPI_Allreduce(&v, &v_max, 1, MPI_DOUBLE, MPI_MAX, adapt_p4est->mpicomm);
  }

  // vorticity
  double vort_temp;
  double vort_min = std::numeric_limits<double>::max();
  double vort_max = std::numeric_limits<double>::min();
  castable_unique_ptr<sc_array_t> vort_values;
  if (p4est_params.threshold_vorticity[0] != 0.0 &&
      p4est_params.threshold_vorticity[1] != 1.0) {
    // max norm
    vort_values =
        sc_array_new_size(sizeof(double), 3 * adapt_p4est->local_num_quadrants);
    lbadapt_get_vorticity_values(vort_values);
    for (int qid = 0; qid < adapt_p4est->local_num_quadrants; ++qid) {
      for (int d = 0; d < P4EST_DIM; ++d) {
        vort_temp = abs(*(double *) sc_array_index(vort_values, 3 * qid + d));
        if (vort_temp < vort_min) {
          vort_min = vort_temp;
        }
        if (vort_temp > vort_max) {
          vort_max = vort_temp;
        }
      }
    }
    // sync
    vort_temp = vort_min;
    MPI_Allreduce(&vort_temp, &vort_min, 1, MPI_DOUBLE, MPI_MIN,
                  adapt_p4est->mpicomm);
    vort_temp = vort_max;
    MPI_Allreduce(&vort_temp, &vort_max, 1, MPI_DOUBLE, MPI_MAX,
                  adapt_p4est->mpicomm);
  }

  // refinement in a potentially moving box
  p8est_quadrant_t *q;
  std::vector<refinement_area_t> refined_patches;
  refinement_area_t ref_area;
  std::array<bool, 3> overlap;
  std::array<double, 3> midpoint;
  std::array<double, 6> default_box =
      {{std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
        std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
        std::numeric_limits<double>::min(), std::numeric_limits<double>::max()}};
  if (coords_for_regional_refinement != default_box) {
    ref_area.bbox_min = {{std::fmod(coords_for_regional_refinement[0] +
                                    sim_time * vel_reg_ref[0], box_l[0]),
                          std::fmod(coords_for_regional_refinement[2] +
                                    sim_time * vel_reg_ref[1], box_l[1]),
                          std::fmod(coords_for_regional_refinement[4] +
                                    sim_time * vel_reg_ref[2], box_l[2])}};
    ref_area.bbox_max = {{std::fmod(coords_for_regional_refinement[1] +
                                    sim_time * vel_reg_ref[0], box_l[0]),
                          std::fmod(coords_for_regional_refinement[3] +
                                    sim_time * vel_reg_ref[1], box_l[1]),
                          std::fmod(coords_for_regional_refinement[5] +
                                    sim_time * vel_reg_ref[2], box_l[2])}};
    refined_patches.push_back(ref_area);
  }

  // particle criterion: Refine cells where we have particles
  double h_max = (0 == sim_time) ? p4est_params.h[p4est_params.min_ref_level]
                                 : p4est_params.h[p4est_params.max_ref_level];
  //double h_max = p4est_params.h[p4est_params.max_ref_level];
  double radius = 2.5 * h_max;
  if (n_part) {
    for (auto p: local_cells.particles()) {
      create_refinement_patch_from_pos(refined_patches, p.r.p, radius);
    }
    for (auto p: ghost_cells.particles()) {
      create_refinement_patch_from_pos(refined_patches, p.r.p, radius);
    }
  }

  if (!refined_patches.empty() ||
      (vel_reg_ref[0] != std::numeric_limits<double>::min() &&
       vel_reg_ref[1] != std::numeric_limits<double>::min() &&
       vel_reg_ref[2] != std::numeric_limits<double>::min()) ||
      (v_min < std::numeric_limits<double>::max()) ||
      (std::numeric_limits<double>::min() < v_max) ||
      (vort_min < std::numeric_limits<double>::max()) ||
      (vort_max < std::numeric_limits<double>::min())) {
    // traverse forest and decide if the current quadrant is to be refined or
    // coarsened
    for (int qid = 0; qid < adapt_p4est->local_num_quadrants; ++qid) {
      // velocity
      if ((v_min < std::numeric_limits<double>::max()) ||
          (std::numeric_limits<double>::min() < v_max)) {
        double v = sqrt(
            Utils::sqr(*(double *) sc_array_index(vel_values, 3 * qid)) +
            Utils::sqr(*(double *) sc_array_index(vel_values, 3 * qid + 1)) +
            Utils::sqr(*(double *) sc_array_index(vel_values, 3 * qid + 2)));
        // Note, that this formulation stems from the fact that velocity is 0 at
        // boundaries
        if (p4est_params.threshold_velocity[1] * (v_max - v_min) <
            (v - v_min)) {
          flags[qid] = 1;
        } else if ((p4est_params.threshold_velocity[0] * (v_max - v_min) <
                    v - v_min) && ((v - v_min) <=
                    p4est_params.threshold_velocity[1] * (v_max - v_min))) {
          flags[qid] = 0;
        }
      }

      // vorticity
      if ((vort_min < std::numeric_limits<double>::max()) ||
          (std::numeric_limits<double>::min() < vort_max)) {
        double vort = std::numeric_limits<double>::min();
        for (int d = 0; d < P4EST_DIM; ++d) {
          vort_temp = abs(*(double *) sc_array_index(vort_values, 3 * qid + d));
          if (vort < vort_temp) {
            vort = vort_temp;
          }
        }
        if (p4est_params.threshold_vorticity[1] * (vort_max - vort_min) <
            (vort - vort_min)) {
          flags[qid] = 1;
        } else if ((p4est_params.threshold_vorticity[0] * (vort_max - vort_min) <
                    vort - vort_min) && ((vort - vort_min) <=
                    p4est_params.threshold_vorticity[1] * (vort_max - vort_min))) {
          flags[qid] = 0;
        }
      }

      // geometry
      for (auto patch: refined_patches) {
        q = p4est_mesh_get_quadrant(adapt_p4est, adapt_mesh, qid);
        p4est_utils_get_midpoint(adapt_p4est, adapt_mesh->quad_to_tree[qid], q,
                                 midpoint.data());

        for (int i = 0; i < 3; ++i) {
          overlap[i] = patch.bbox_min[i] > patch.bbox_max[i];
        }

        // support boundary moving out of domain as well:
        // 3 comparisons:
        // (min - box_l) < c && c < max -> wrap min
        // min < c && c < (max + box_l) -> wrap max
        // min < c && c < max           -> standard
        if (((!overlap[0] &&
              (patch.bbox_min[0] < midpoint[0] &&
               midpoint[0] < patch.bbox_max[0])) ||
             (overlap[0] &&
              (midpoint[0] < patch.bbox_max[0] ||
               patch.bbox_min[0] < midpoint[0]))) &&
            ((!overlap[1] &&
              (patch.bbox_min[1] < midpoint[1] &&
               midpoint[1] < patch.bbox_max[1])) ||
             (overlap[1] &&
              (midpoint[1] < patch.bbox_max[1] ||
               patch.bbox_min[1] < midpoint[1]))) &&
            ((!overlap[2] &&
              (patch.bbox_min[2] < midpoint[2] &&
               midpoint[2] < patch.bbox_max[2])) ||
             (overlap[2] &&
              (midpoint[2] < patch.bbox_max[2] ||
               patch.bbox_min[2] < midpoint[2])))) {
          flags[qid] = 1;
        }
      }
    }
  }
#endif // LB_ADAPTIVE
}

/** Dummy initialization function for quadrants created in refinement step
 */
void p4est_utils_qid_dummy (p8est_t *p8est, p4est_topidx_t which_tree,
                            p8est_quadrant_t *q) {
  q->p.user_long = -1;
}

int p4est_utils_end_pending_communication(
    std::vector<p8est_virtual_ghost_exchange_t*> &exc_status, int level) {
#ifdef LB_ADAPTIVE
#ifdef COMM_HIDING
  if (-1 == level) {
    for (int i = p4est_params.min_ref_level;
         i < p4est_params.max_ref_level; ++i) {
      if (nullptr != exc_status[i]) {
        p4est_virtual_ghost_exchange_data_level_end(exc_status[i]);
        exc_status[i] = nullptr;
      }
    }
  } else {
    if (nullptr != exc_status[level]) {
      p4est_virtual_ghost_exchange_data_level_end(exc_status[level]);
      exc_status[level] = nullptr;
    }
  }
#endif // COMM_HIDING
#endif // LB_ADAPTIVE
  return 0;
}

int p4est_utils_perform_adaptivity_step() {
#ifdef LB_ADAPTIVE
  p4est_connect_type_t btype = P4EST_CONNECT_FULL;

  // 1st step: alter copied grid and map data between grids.
  // collect refinement and coarsening flags.
  flags = std::vector<int>(adapt_p4est->local_num_quadrants, 2);
  p4est_utils_collect_flags(flags);

  // To guarantee that we can map quadrants probably to their qids write each
  // quadrant's qid into a payload (this is needed due to p4est relying on
  // mempool by libsc).
  p4est_iterate(adapt_p4est, adapt_ghost, nullptr, lbadapt_init_qid_payload,
                nullptr, nullptr, nullptr);

#ifdef COMM_HIDING
  // get rid of any pending communication and avoid dangling messages
  p4est_utils_end_pending_communication(exc_status_lb);
#endif

  // copy forest and perform refinement step.
  p8est_t *p4est_adapted = p8est_copy(adapt_p4est, 0);
  P4EST_ASSERT(p4est_is_equal(p4est_adapted, adapt_p4est, 0));
  p8est_refine_ext(p4est_adapted, 0, p4est_params.max_ref_level,
                   refinement_criteria, p4est_utils_qid_dummy, nullptr);
  // perform coarsening step
  p8est_coarsen_ext(p4est_adapted, 0, 0, coarsening_criteria, nullptr, nullptr);
  flags.clear();
  // balance forest after grid change
  p8est_balance_ext(p4est_adapted, P8EST_CONNECT_FULL, nullptr, nullptr);

  // 2nd step: locally map data between forests.
  // de-allocate invalid storage and data-structures
  p4est_utils_deallocate_levelwise_storage(lbadapt_ghost_data);
  adapt_virtual_ghost.reset();
  adapt_ghost.reset();

  // locally map data between forests.
  linear_payload_lbm.resize(p4est_adapted->local_num_quadrants);

  p4est_utils_post_gridadapt_map_data(adapt_p4est, adapt_mesh, adapt_virtual,
                                      p4est_adapted, lbadapt_local_data,
                                      linear_payload_lbm.data());

  // cleanup
  p4est_utils_deallocate_levelwise_storage(lbadapt_local_data);
  adapt_p4est.reset(p4est_adapted);
  adapt_virtual.reset();
  adapt_mesh.reset();

  // 3rd step: partition grid and transfer data to respective new owner ranks
  //           including all preparations for next time step
#ifdef DD_P4EST
  p4est_utils_prepare({dd_p4est_get_p4est(), adapt_p4est});

  std::vector<std::string> metrics = {"ncells", p4est_params.partitioning};
  std::vector<double> alpha = {1., 1.};
  lbmd::repart_all(metrics, alpha);
#else
  p4est_utils_repart_preprocess();
  if (p4est_params.partitioning == "n_cells") {
    p8est_partition_ext(adapt_p4est, 1, lbadapt_partition_weight_uniform);
  }
  else if (p4est_params.partitioning == "subcycling") {
    p8est_partition_ext(adapt_p4est, 1, lbadapt_partition_weight_subcycling);
  }
  else {
    SC_ABORT_NOT_REACHED();
  }
  p4est_utils_repart_postprocess();
#endif
#endif // LB_ADAPTIVE
  return 0;
}

template <typename T>
int p4est_utils_post_gridadapt_map_data(
    p4est_t *p4est_old, p4est_mesh_t *mesh_old, p4est_virtual_t *virtual_quads,
    p4est_t *p8est_new, std::vector<std::vector<T>> &local_data_levelwise,
    T *mapped_data_flat) {
  // counters
  unsigned int tid_old = p4est_old->first_local_tree;
  unsigned int tid_new = p4est_new->first_local_tree;
  unsigned int qid_old = 0, qid_new = 0;
  unsigned int tqid_old = 0, tqid_new = 0;

  // trees
  p8est_tree_t *curr_tree_old =
      p8est_tree_array_index(p4est_old->trees, tid_old);
  p8est_tree_t *curr_tree_new =
      p8est_tree_array_index(p4est_new->trees, tid_new);
  // quadrants
  p8est_quadrant_t *curr_quad_old, *curr_quad_new;

  int level_old, sid_old;
  int level_new;

  while (qid_old < (size_t) p4est_old->local_num_quadrants &&
         qid_new < (size_t) p4est_new->local_num_quadrants) {
    // wrap multiple trees
    if (tqid_old == curr_tree_old->quadrants.elem_count) {
      ++tid_old;
      P4EST_ASSERT(tid_old < p4est_old->trees->elem_count);
      curr_tree_old = p8est_tree_array_index(p4est_old->trees, tid_old);
      tqid_old = 0;
    }
    if (tqid_new == curr_tree_new->quadrants.elem_count) {
      ++tid_new;
      P4EST_ASSERT(tid_new < p4est_new->trees->elem_count);
      curr_tree_new = p8est_tree_array_index(p4est_new->trees, tid_new);
      tqid_new = 0;
    }

    // fetch next quadrants in old and new forest and obtain storage id
    curr_quad_old =
        p8est_quadrant_array_index(&curr_tree_old->quadrants, tqid_old);
    level_old = curr_quad_old->level;
    sid_old = virtual_quads->quad_qreal_offset[qid_old];

    curr_quad_new =
        p8est_quadrant_array_index(&curr_tree_new->quadrants, tqid_new);
    level_new = curr_quad_new->level;

    // distinguish three cases to properly map data and increase indices
    if (level_old == level_new) {
      // old cell has neither been coarsened nor refined
      data_transfer(p4est_old, p4est_new, curr_quad_old, curr_quad_new, tid_old,
                    &local_data_levelwise[level_old][sid_old],
                    &mapped_data_flat[qid_new]);
      ++qid_old;
      ++qid_new;
      ++tqid_old;
      ++tqid_new;
    } else if (level_old == level_new + 1) {
      // old cell has been coarsened
      for (int child = 0; child < P8EST_CHILDREN; ++child) {
        data_restriction(p4est_old, p4est_new, curr_quad_old, curr_quad_new,
                         tid_old, &local_data_levelwise[level_old][sid_old],
                         &mapped_data_flat[qid_new]);
        ++sid_old;
        ++tqid_old;
        ++qid_old;
      }
      ++tqid_new;
      ++qid_new;
    } else if (level_old + 1 == level_new) {
      // old cell has been refined.
      for (int child = 0; child < P8EST_CHILDREN; ++child) {
        data_interpolation(p4est_old, p4est_new, curr_quad_old, curr_quad_new,
                           tid_old, &local_data_levelwise[level_old][sid_old],
                           &mapped_data_flat[qid_new]);
        ++tqid_new;
        ++qid_new;
      }
      ++tqid_old;
      ++qid_old;
    } else {
      SC_ABORT_NOT_REACHED();
    }

    // sanity check of indices
    P4EST_ASSERT(tqid_old + curr_tree_old->quadrants_offset == qid_old);
    P4EST_ASSERT(tqid_new + curr_tree_new->quadrants_offset == qid_new);
    P4EST_ASSERT(tid_old == tid_new);
  }
  P4EST_ASSERT(qid_old == (size_t) p4est_old->local_num_quadrants);
  P4EST_ASSERT(qid_new == (size_t) p4est_new->local_num_quadrants);

  return 0;
}

void get_subc_weights(p8est_iter_volume_info_t *info, void *user_data) {
  std::vector<double> *w = reinterpret_cast<std::vector<double>* >(user_data);
  w->at(info->quadid) = 1 << (p4est_params.max_ref_level - info->quad->level);
}


#if defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)
std::vector<double> p4est_utils_get_adapt_weights(const std::string& metric) {
  std::vector<double> weights (adapt_p4est->local_num_quadrants, 1.0);
  if (metric == "subcycling") {
    p4est_iterate(adapt_p4est, nullptr, &weights, get_subc_weights, nullptr,
                  nullptr, nullptr);
  } else if (metric != "n_cells") {
    fprintf(stderr, "Unknown metric: %s\n", metric.c_str());
    errexit();
  }
  return weights;
}

int p4est_utils_repart_preprocess() {
  if (linear_payload_lbm.empty()) {
    linear_payload_lbm.resize(adapt_p4est->local_num_quadrants);
    p4est_utils_flatten_data(adapt_p4est, adapt_mesh, adapt_virtual,
                             lbadapt_local_data, linear_payload_lbm);
  }
  // Save global_first_quadrants for migration
  old_partition_table_adapt.clear();
  std::copy_n(adapt_p4est->global_first_quadrant, n_nodes + 1,
              std::back_inserter(old_partition_table_adapt));
  return 0;
}

int p4est_utils_repart_postprocess() {
  // transfer payload
  std::vector<lbadapt_payload_t> recv_buffer;
  recv_buffer.resize(adapt_p4est->local_num_quadrants);

#ifdef COMM_HIDING
  auto data_transfer_handle = p4est_transfer_fixed_begin(
      adapt_p4est->global_first_quadrant, old_partition_table_adapt.data(), comm_cart,
      3172 + sizeof(lbadapt_payload_t), recv_buffer.data(),
      linear_payload_lbm.data(), sizeof(lbadapt_payload_t));
#else  // COMM_HIDING
  p4est_transfer_fixed(adapt_p4est->global_first_quadrant,
                       old_partition_table_adapt.data(), comm_cart,
                       3172 + sizeof(lbadapt_payload_t), recv_buffer.data(),
                       linear_payload_lbm.data(), sizeof(lbadapt_payload_t));
#endif // COMM_HIDING

  // recreate p4est structs after partitioning
  p4est_utils_rebuild_p4est_structs(P8EST_CONNECT_FULL, false);

#ifdef COMM_HIDING
  p4est_transfer_fixed_end(data_transfer_handle);
#endif // COMM_HIDING

  // clear linear data-structure
  linear_payload_lbm.clear();

  // allocate data
  p4est_utils_allocate_levelwise_storage(lbadapt_local_data, adapt_mesh,
                                         adapt_virtual, true);

  if (adapt_ghost->ghosts.elem_count) {
    p4est_utils_allocate_levelwise_storage(lbadapt_ghost_data, adapt_mesh,
                                           adapt_virtual, false);
  }

  // insert data in per-level data-structure
  p4est_utils_unflatten_data(adapt_p4est, adapt_mesh, adapt_virtual,
                             recv_buffer, lbadapt_local_data);

  // synchronize ghost data for next collision step
  std::vector<lbadapt_payload_t *> local_pointer(P8EST_QMAXLEVEL);
  std::vector<lbadapt_payload_t *> ghost_pointer(P8EST_QMAXLEVEL);
  prepare_ghost_exchange(lbadapt_local_data, local_pointer,
                         lbadapt_ghost_data, ghost_pointer);
  for (int level = p4est_params.min_ref_level;
       level <= p4est_params.max_ref_level; ++level) {
#ifdef COMM_HIDING
    exc_status_lb[level] =
        p4est_virtual_ghost_exchange_data_level_begin(
            adapt_p4est, adapt_ghost, adapt_mesh, adapt_virtual,
            adapt_virtual_ghost, level, sizeof(lbadapt_payload_t),
            (void**)local_pointer.data(), (void**)ghost_pointer.data());
#else // COMM_HIDING
    p4est_virtual_ghost_exchange_data_level (adapt_p4est, adapt_ghost,
                                             adapt_mesh, adapt_virtual,
                                             adapt_virtual_ghost, level,
                                             sizeof(lbadapt_payload_t),
                                             (void**)local_pointer.data(),
                                             (void**)ghost_pointer.data());
#endif // COMM_HIDING
  }
  p4est_utils_prepare({dd_p4est_get_p4est(), adapt_p4est});

  return 0;
}
#endif // defined(LB_ADAPTIVE) || defined(ES_ADAPTIVE) || defined(EK_ADAPTIVE)

void p4est_utils_partition_multiple_forests(forest_order reference,
                                            forest_order modify) {
#if defined(DD_P4EST) && defined (LB_ADAPTIVE)
  p8est_t *p4est_ref = forest_info.at(static_cast<int>(reference)).p4est;
  p8est_t *p4est_mod = forest_info.at(static_cast<int>(modify)).p4est;
  if (p4est_ref && p4est_mod) {
    P4EST_ASSERT(p4est_ref->mpisize == p4est_mod->mpisize);
    P4EST_ASSERT(p4est_ref->mpirank == p4est_mod->mpirank);
    P4EST_ASSERT(p8est_connectivity_is_equivalent(p4est_ref->connectivity,
                                                  p4est_mod->connectivity));

    std::vector<p4est_locidx_t> num_quad_per_proc(p4est_ref->mpisize, 0);
    std::vector<p4est_locidx_t> num_quad_per_proc_global(p4est_ref->mpisize, 0);

    unsigned int tid = p4est_mod->first_local_tree;
    unsigned int tqid = 0;
    // trees
    p8est_tree_t *curr_tree;
    // quadrants
    p8est_quadrant_t *curr_quad;

    if (0 < p4est_mod->local_num_quadrants) {
      curr_tree = p8est_tree_array_index(p4est_mod->trees, tid);
    }

    // Check for each of the quadrants of the given p4est, to which cell it maps
    // in the other forest
    for (int qid = 0; qid < p4est_mod->local_num_quadrants; ++qid) {
      // wrap multiple trees
      if (tqid == curr_tree->quadrants.elem_count) {
        ++tid;
        P4EST_ASSERT(tid < p4est_mod->trees->elem_count);
        curr_tree = p8est_tree_array_index(p4est_mod->trees, tid);
        tqid = 0;
      }
      if (0 < curr_tree->quadrants.elem_count) {
        curr_quad = p8est_quadrant_array_index(&curr_tree->quadrants, tqid);
        double xyz[3];
        p4est_utils_get_front_lower_left(p4est_mod, tid, curr_quad, xyz);
        int proc = p4est_utils_pos_to_proc(reference, xyz);
        ++num_quad_per_proc[proc];
      }
      ++tqid;
    }

    // Gather this information over all processes
    MPI_Allreduce(num_quad_per_proc.data(), num_quad_per_proc_global.data(),
                  p4est_mod->mpisize, P4EST_MPI_LOCIDX, MPI_SUM,
                  p4est_mod->mpicomm);

    p4est_locidx_t sum = std::accumulate(std::begin(num_quad_per_proc_global),
                                         std::end(num_quad_per_proc_global), 0);

    if (sum < p4est_mod->global_num_quadrants) {
      fprintf(stderr, "%i : %li quadrants lost while partitioning\n",
              this_node, p4est_mod->global_num_quadrants - sum);
      errexit();
    }

    CELL_TRACE(printf("%i : repartitioned LB %i\n", this_node,
                      num_quad_per_proc_global[this_node]));

    // Repartition with the computed distribution
    int shipped =
        p8est_partition_given(p4est_mod, num_quad_per_proc_global.data());
    P4EST_GLOBAL_PRODUCTIONF(
        "Done " P8EST_STRING "_partition shipped %lld quadrants %.3g%%\n",
        (long long) shipped, shipped * 100. / p4est_mod->global_num_quadrants);
  } else {
    p4est_t * existing_forest;
    if (p4est_mod) {
      existing_forest = p4est_mod;
    }
    else {
      existing_forest = p4est_ref;
    }
    if (p4est_params.partitioning == "n_cells") {
      p8est_partition_ext(existing_forest, 1,
                          lbadapt_partition_weight_uniform);
    }
    else if (p4est_params.partitioning == "subcycling") {
      p8est_partition_ext(existing_forest, 1,
                          lbadapt_partition_weight_subcycling);
    }
    else {
      SC_ABORT_NOT_REACHED();
    }

    if (this_node == 0) {
      std::cerr
          << "Not all p4ests have been created yet. This may happen during"
          << " initialization." << std::endl;
    }
  }
#endif // defined(DD_P4EST) && defined(LB_ADAPTIVE)
}

static int fct_coarsen_cb(p4est_t *p4est, p4est_topidx_t tree_idx,
                   p4est_quadrant_t *quad[]) {
  p4est_t *cmp = (p4est_t *)p4est->user_pointer;
  p4est_tree_t *tree = p4est_tree_array_index(cmp->trees, tree_idx);
  for (unsigned int i = 0; i < tree->quadrants.elem_count; ++i) {
    p4est_quadrant_t *q = p4est_quadrant_array_index(&tree->quadrants, i);
    if (p4est_quadrant_overlaps(q, quad[0]))
      return q->level < quad[0]->level;
  }

  // We have to find at least one overlapping quadrant in the above loop
  SC_ABORT_NOT_REACHED();
}

p4est_t *p4est_utils_create_fct(p4est_t *t1, p4est_t *t2) {
  p4est_t *fct = p4est_copy(t2, 0);
  fct->user_pointer = (void *)t1;
  p4est_coarsen(fct, 1, fct_coarsen_cb, NULL);
  return fct;
}

bool p4est_utils_check_alignment(const p4est_t *t1, const p4est_t *t2) {
  if (!p4est_connectivity_is_equivalent(t1->connectivity, t2->connectivity))
    return false;
  if (t1->first_local_tree != t2->first_local_tree) return false;
  if (t1->last_local_tree != t2->last_local_tree) return false;
  p4est_quadrant_t *q1 = &t1->global_first_position[t1->mpirank];
  p4est_quadrant_t *q2 = &t2->global_first_position[t2->mpirank];
  if (!p4est_quadrant_is_equal(q1, q2)) return false;
  q1 = &t1->global_first_position[t1->mpirank+1];
  q2 = &t2->global_first_position[t2->mpirank+1];
  if (!p4est_quadrant_is_equal(q1, q2)) return false;
  return true;
}

void p4est_utils_weighted_partition(p4est_t *t1, const std::vector<double> &w1,
                                    double a1, p4est_t *t2,
                                    const std::vector<double> &w2, double a2) {
  P4EST_ASSERT(w1.size() == t1->local_num_quadrants);
  P4EST_ASSERT(w2.size() == t2->local_num_quadrants);
  P4EST_ASSERT(p4est_utils_check_alignment(t1, t2));

  std::unique_ptr<p4est_t> fct(p4est_utils_create_fct(t1, t2));
  P4EST_ASSERT(p4est_is_valid(fct.get()));
  P4EST_ASSERT(p4est_utils_check_alignment(fct.get(), t1));

  std::vector<double> w_fct(fct->local_num_quadrants, 0.0);

  std::vector<size_t> t1_quads_per_fct_quad(fct->local_num_quadrants, 0);
  std::vector<size_t> t2_quads_per_fct_quad(fct->local_num_quadrants, 0);
  std::vector<p4est_locidx_t> t1_quads_per_proc(fct->mpisize, 0);
  std::vector<p4est_locidx_t> t2_quads_per_proc(fct->mpisize, 0);

  size_t w_id1, w_id2, w_idx;
  w_id1 = w_id2 = w_idx = 0;
  for (p4est_topidx_t t_idx = fct->first_local_tree;
       t_idx <= fct->last_local_tree; ++t_idx) {
    p4est_tree_t *t_fct = p4est_tree_array_index(fct->trees, t_idx);
    p4est_tree_t *t_t1  = p4est_tree_array_index(t1->trees, t_idx);
    p4est_tree_t *t_t2  = p4est_tree_array_index(t2->trees, t_idx);
    size_t q_id1, q_id2;
    q_id1 = q_id2 = 0;
    p4est_quadrant_t *q1 = p4est_quadrant_array_index(&t_t1->quadrants, q_id1);
    p4est_quadrant_t *q2 = p4est_quadrant_array_index(&t_t2->quadrants, q_id2);
    for (size_t q_idx = 0; q_idx < t_fct->quadrants.elem_count; ++q_idx) {
      p4est_quadrant_t *q_fct =
          p4est_quadrant_array_index(&t_fct->quadrants, q_idx);
      while (p4est_quadrant_overlaps(q_fct, q1)) {
        w_fct[w_idx] += a1 * w1[w_id1++];
        ++t1_quads_per_fct_quad[w_idx];
        if (++q_id1 >= t_t1->quadrants.elem_count) {
          // complain if last quad in t1 does not overlap with last quad of FCT
          P4EST_ASSERT(q_idx == t_fct->quadrants.elem_count - 1);
          break;
        }
        q1 = p4est_quadrant_array_index(&t_t1->quadrants, q_id1);
      }
      while (p4est_quadrant_overlaps(q_fct, q2)) {
        w_fct[w_idx] += a2 * w2[w_id2++];
        ++t2_quads_per_fct_quad[w_idx];
        if (++q_id2 >= t_t2->quadrants.elem_count) {
          // complain if last quad in t2 does not overlap with last quad of FCT
          P4EST_ASSERT(q_idx == t_fct->quadrants.elem_count - 1);
          break;
        }
        q2 = p4est_quadrant_array_index(&t_t2->quadrants, q_id2);
      }
      ++w_idx;
    }
  }

  // complain if counters haven't reached the end
  P4EST_ASSERT(w_idx == (size_t) fct->local_num_quadrants);
  P4EST_ASSERT(w_id1 == (size_t) t1->local_num_quadrants);
  P4EST_ASSERT(w_id2 == (size_t) t2->local_num_quadrants);

  double localsum = std::accumulate(w_fct.begin(), w_fct.end(), 0.0);
  double sum, prefix = 0; // Initialization is necessary on rank 0!
  MPI_Allreduce(&localsum, &sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
  MPI_Exscan(&localsum, &prefix, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
  double target = sum / fct->mpisize;

  for (size_t idx = 0; idx < (size_t) fct->local_num_quadrants; ++idx) {
    double old_prefix = prefix;
    prefix += w_fct[idx];
    int proc = std::min<int>(0.5 * (prefix + old_prefix) / target, fct->mpisize - 1);
    t1_quads_per_proc[proc] += t1_quads_per_fct_quad[idx];
    t2_quads_per_proc[proc] += t2_quads_per_fct_quad[idx];
  }

  MPI_Allreduce(MPI_IN_PLACE, t1_quads_per_proc.data(), fct->mpisize,
                P4EST_MPI_LOCIDX, MPI_SUM, comm_cart);
  MPI_Allreduce(MPI_IN_PLACE, t2_quads_per_proc.data(), fct->mpisize,
                P4EST_MPI_LOCIDX, MPI_SUM, comm_cart);

  p8est_partition_given(t1, t1_quads_per_proc.data());
  p8est_partition_given(t2, t2_quads_per_proc.data());
}

bool adaptive_lb_is_active()
{
#ifdef LB_ADAPTIVE
  return adapt_p4est != nullptr;
#else
  return false;
#endif
}

#endif // defined (LB_ADAPTIVE) || defined (DD_P4EST)
