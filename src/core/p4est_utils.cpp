#include "p4est_utils.hpp"

#if (defined(LB_ADAPTIVE) || defined(DD_P4EST))

#include "debug.hpp"
#include "domain_decomposition.hpp"
#include "lb-adaptive.hpp"
#include "p4est_dd.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mpi.h>
#include <p8est_algorithms.h>
#include <p8est_bits.h>
#include <p8est_search.h>
#include <vector>

static std::vector<p4est_utils_forest_info_t> forest_info;

// number of (MD) intergration steps before grid changes
int steps_until_grid_change = 50;

const p4est_utils_forest_info_t &p4est_utils_get_forest_info(forest_order fo) {
  // Use at() here because forest_info might not have been initialized yet.
  return forest_info.at(static_cast<int>(fo));
}

static inline void tree_to_boxlcoords(double x[3]) {
  for (int i = 0; i < 3; ++i)
    x[i] *= box_l[i] / dd_p4est_num_trees_in_dir(i);
}

static inline void maybe_tree_to_boxlcoords(double x[3]) {
#ifndef LB_ADAPTIVE
  tree_to_boxlcoords(x);
#else
  // Id mapping
#endif
}

static inline void boxl_to_treecoords(double x[3]) {
  for (int i = 0; i < 3; ++i)
    x[i] /= (box_l[i] / dd_p4est_num_trees_in_dir(i));
}

static inline void maybe_boxl_to_treecoords(double x[3]) {
#ifndef LB_ADAPTIVE
  boxl_to_treecoords(x);
#else
   // Id mapping
#endif
}

static inline std::array<double, 3>
maybe_boxl_to_treecoords_copy(const double x[3]) {
  std::array<double, 3> res{{x[0], x[1], x[2]}};
  maybe_boxl_to_treecoords(res.data());
  return res;
}

// forward declaration
int64_t
p4est_utils_pos_morton_idx_global(p8est_t *p4est, int level,
                                  std::vector<int> tree_quadrant_offset_synced,
                                  const double pos[3]);

static p4est_utils_forest_info_t p4est_to_forest_info(p4est_t *p4est) {
  // fill element to insert
  p4est_utils_forest_info_t insert_elem(p4est);

  // allocate a local send buffer to insert local quadrant offsets
  std::vector<p4est_locidx_t> local_tree_offsets(p4est->trees->elem_count);

  // fetch last tree index from last processor
  p4est_topidx_t last_tree_prev_rank = -1;
  if (p4est->mpirank != p4est->mpisize - 1) {
    MPI_Send(&p4est->last_local_tree, 1, P4EST_MPI_TOPIDX, p4est->mpirank + 1,
             p4est->mpirank, p4est->mpicomm);
  }
  if (p4est->mpirank != 0) {
    MPI_Recv(&last_tree_prev_rank, 1, P4EST_MPI_TOPIDX, p4est->mpirank - 1,
             p4est->mpirank - 1, p4est->mpicomm, MPI_STATUS_IGNORE);
  }
  // only fill local send buffer if current process is not empty
  if (p4est->local_num_quadrants != 0) {
    // set start index; if first tree is not completely owned by current
    // process it will set a wrong quadrant offset
    int start_idx = (p4est->first_local_tree == last_tree_prev_rank)
                        ? p4est->first_local_tree + 1
                        : p4est->first_local_tree;
    for (int i = p4est->first_local_tree; i <= p4est->last_local_tree; ++i) {
      p8est_tree_t *tree = p8est_tree_array_index(p4est->trees, i);
      if (start_idx <= i) {
        local_tree_offsets[i] = tree->quadrants_offset +
                                p4est->global_first_quadrant[p4est->mpirank];
      }
      /* local max level */
      if (insert_elem.finest_level_local < tree->maxlevel) {
        insert_elem.finest_level_local = insert_elem.coarsest_level_local =
            tree->maxlevel;
      }
      /* local min level */
      for (int l = insert_elem.coarsest_level_local; l >= 0; --l) {
        if (l < insert_elem.coarsest_level_local &&
            tree->quadrants_per_level[l]) {
          insert_elem.coarsest_level_local = l;
        }
      }
    }
  }
  // synchronize offsets and level and insert into forest_info vector
  // clang-format off
  MPI_Allreduce(local_tree_offsets.data(),
                insert_elem.tree_quadrant_offset_synced.data(),
                p4est->trees->elem_count, P4EST_MPI_LOCIDX, MPI_MAX,
                p4est->mpicomm);
  // clang-format on
  MPI_Allreduce(&insert_elem.finest_level_local,
                &insert_elem.finest_level_global, 1, P4EST_MPI_LOCIDX, MPI_MAX,
                p4est->mpicomm);
  MPI_Allreduce(&insert_elem.coarsest_level_local,
                &insert_elem.coarsest_level_global, 1, P4EST_MPI_LOCIDX,
                MPI_MIN, p4est->mpicomm);
  insert_elem.finest_level_ghost = insert_elem.finest_level_global;
  insert_elem.coarsest_level_ghost = insert_elem.coarsest_level_global;

  // ensure monotony
  P4EST_ASSERT(std::is_sorted(insert_elem.tree_quadrant_offset_synced.begin(),
                              insert_elem.tree_quadrant_offset_synced.end()));

  for (int i = 0; i < p4est->mpisize; ++i) {
    p4est_quadrant_t *q = &p4est->global_first_position[i];
    double xyz[3];
    p4est_utils_get_front_lower_left(p4est, q->p.which_tree, q, xyz);

    // Scale xyz because p4est_utils_pos_morton_idx_global will assume it is
    // and undo this.
    maybe_tree_to_boxlcoords(xyz);

    insert_elem.first_quad_morton_idx[i] = p4est_utils_pos_morton_idx_global(
        p4est, insert_elem.finest_level_global,
        insert_elem.tree_quadrant_offset_synced, xyz);
  }
  insert_elem.first_quad_morton_idx[p4est->mpisize] =
      p4est->trees->elem_count *
      (1 << (P8EST_DIM * insert_elem.finest_level_global));
  P4EST_ASSERT(std::is_sorted(insert_elem.first_quad_morton_idx.begin(),
                              insert_elem.first_quad_morton_idx.end()));

  return insert_elem;
}

void p4est_utils_prepare(std::vector<p8est_t *> p4ests) {
  forest_info.clear();

  std::transform(std::begin(p4ests), std::end(p4ests),
                 std::back_inserter(forest_info), p4est_to_forest_info);
}

int p4est_utils_pos_to_proc(forest_order forest, const double pos[3]) {
  const p4est_utils_forest_info_t &current_forest =
      forest_info.at(static_cast<int>(forest));
  int qid = p4est_utils_pos_morton_idx_global(forest, pos);

  int p = std::distance(current_forest.first_quad_morton_idx.begin(),
                        std::upper_bound(
                            current_forest.first_quad_morton_idx.begin(),
                            current_forest.first_quad_morton_idx.end(), qid)) -
          1;

  P4EST_ASSERT(0 <= p && p < current_forest.p4est->mpisize);

  return p;
}

int64_t p4est_utils_cell_morton_idx(int x, int y, int z) {
  int64_t idx = 0;
  int64_t pos = 1;

  for (int i = 0; i < 21; ++i) {
    if ((x & 1))
      idx += pos;
    x >>= 1;
    pos <<= 1;
    if ((y & 1))
      idx += pos;
    y >>= 1;
    pos <<= 1;
    if ((z & 1))
      idx += pos;
    z >>= 1;
    pos <<= 1;
  }

  return idx;
}

/**
 * CAUTION: If LB_ADAPTIVE is not set, all p4ests will be scaled by the side
 *          length of the p4est instance used for short-ranged MD.
 */
static int p4est_utils_map_pos_to_tree(p4est_t *p4est, const double pos[3]) {
  int tid = -1;
  for (int t = 0; t < p4est->connectivity->num_trees; ++t) {
    // collect corners of tree
    std::array<double, 3> c[P4EST_CHILDREN];
    for (int ci = 0; ci < P4EST_CHILDREN; ++ci) {
      int v = p4est->connectivity->tree_to_vertex[t * P4EST_CHILDREN + ci];
      c[ci][0] = p4est->connectivity->vertices[P4EST_DIM * v + 0];
      c[ci][1] = p4est->connectivity->vertices[P4EST_DIM * v + 1];
      c[ci][2] = p4est->connectivity->vertices[P4EST_DIM * v + 2];

      // As pure MD allows for box_l != 1.0, "pos" will be in [0,box_l) and
      // not in [0,1). So manually scale the trees to fill [0,box_l).
      maybe_tree_to_boxlcoords(c[ci].data());
    }

    // find lower left and upper right corner of forest
    std::array<double, 3> pos_min{{0., 0., 0.}};
    std::array<double, 3> pos_max{{box_l[0], box_l[1], box_l[2]}};
    int idx_min, idx_max;
    double dist;
    double dist_min = DBL_MAX;
    double dist_max = DBL_MAX;
    for (int ci = 0; ci < P4EST_CHILDREN; ++ci) {
      dist = distance(c[ci], pos_min);
      if (dist < dist_min) {
        dist_min = dist;
        idx_min = ci;
      }
      dist = distance(c[ci], pos_max);
      if (dist < dist_max) {
        dist_max = dist;
        idx_max = ci;
      }
    }

    // if position is between lower left and upper right corner of forest this
    // is the right tree
    if ((c[idx_min][0] <= pos[0]) && (c[idx_min][1] <= pos[1]) &&
        (c[idx_min][2] <= pos[2]) && (pos[0] < c[idx_max][0]) &&
        (pos[1] < c[idx_max][1]) && (pos[2] < c[idx_max][2])) {
      // ensure trees do not overlap
      P4EST_ASSERT(-1 == tid);
      tid = t;
    }
  }
  // ensure that we found a tree
  P4EST_ASSERT(tid != -1);
  return tid;
}

int64_t
p4est_utils_pos_morton_idx_global(p8est_t *p4est, int level,
                                  std::vector<int> tree_quadrant_offset_synced,
                                  const double pos[3]) {

  // find correct tree
  int tid = p4est_utils_map_pos_to_tree(p4est, pos);
  // Qpos is the 3d cell index within tree "tid".
  int qpos[3];

  // In case of pure MD arbitrary numbers are allowed for box_l.
  // Scale "spos" such that it corresponds to a box_l of 1.0
  auto spos = maybe_boxl_to_treecoords_copy(pos);

  int nq = 1 << level;
  for (int i = 0; i < P8EST_DIM; ++i) {
    qpos[i] = (spos[i] - (int)spos[i]) * nq;
    P4EST_ASSERT(0 <= qpos[i] && qpos[i] < nq);
  }

  int qid = p4est_utils_cell_morton_idx(qpos[0], qpos[1], qpos[2]) +
            tree_quadrant_offset_synced[tid];

  return qid;
}

int64_t p4est_utils_pos_morton_idx_global(forest_order forest,
                                          const double pos[3]) {
  const p4est_utils_forest_info_t &current_p4est =
      forest_info.at(static_cast<int>(forest));
  return p4est_utils_pos_morton_idx_global(
      current_p4est.p4est, current_p4est.finest_level_global,
      current_p4est.tree_quadrant_offset_synced, pos);
}

static inline bool is_valid_local_quad(const p8est *p4est, int64_t quad) {
  return quad >= 0 && quad < p4est->local_num_quadrants;
}

#define RETURN_IF_VALID_QUAD(q, fo)                                            \
  do {                                                                         \
    int64_t qid = q;                                                           \
    if (is_valid_local_quad(forest_info[static_cast<int>(fo)].p4est, qid))     \
      return qid;                                                              \
  } while (0)

int64_t p4est_utils_pos_quad_ext(forest_order forest, const double pos[3]) {
  // Try pos itself
  RETURN_IF_VALID_QUAD(p4est_utils_pos_qid_local(forest, pos), forest);

  // If pos is outside of the local domain try the bounding box enlarged
  // ROUND_ERROR_PREC
  for (int i = -1; i <= 1; i += 2) {
    for (int j = -1; j <= 1; j += 2) {
      for (int k = -1; k <= 1; k += 2) {
        double spos[3] = {pos[0] + i * box_l[0] * ROUND_ERROR_PREC,
                          pos[1] + j * box_l[1] * ROUND_ERROR_PREC,
                          pos[2] + k * box_l[2] * ROUND_ERROR_PREC};

        RETURN_IF_VALID_QUAD(p4est_utils_pos_qid_local(forest, spos), forest);
      }
    }
  }

  return -1;
}

static int p4est_utils_find_qid_prepare(forest_order forest,
                                        const double pos[3],
                                        p8est_tree_t **tree,
                                        p8est_quadrant_t *pquad) {
  const p4est_utils_forest_info_t &current_p4est =
      forest_info.at(static_cast<int>(forest));
  p8est_t *p4est = current_p4est.p4est;

  // find correct tree
  int tid = p4est_utils_map_pos_to_tree(p4est, pos);
  int level = current_p4est.finest_level_global;
  *tree = p4est_tree_array_index(p4est->trees, tid);

  double first_pos[3];
  p4est_qcoord_to_vertex(p4est->connectivity, tid, 0, 0, 0, first_pos);

  // Trees might not have a base length of 1
  auto spos = maybe_boxl_to_treecoords_copy(pos);

  int qcoord[3];
  for (int i = 0; i < P8EST_DIM; ++i) {
    qcoord[i] = (spos[i] - first_pos[i]) * (1 << level);
  }

  int64_t pidx = p4est_utils_cell_morton_idx(qcoord[0], qcoord[1], qcoord[2]);
  p4est_quadrant_set_morton(pquad, level, pidx);
  pquad->p.which_tree = tid;

  return 0;
}

p4est_locidx_t p4est_utils_pos_qid_local(forest_order forest,
                                         const double pos[3]) {
  p4est_tree_t *tree;
  p4est_quadrant_t pquad;
  p4est_utils_find_qid_prepare(forest, pos, &tree, &pquad);

  p4est_locidx_t index = p8est_find_lower_bound_overlap(
      &tree->quadrants, &pquad, 0.5 * tree->quadrants.elem_count);

#ifdef P4EST_ENABLE_DEBUG
  p8est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, index);
  P4EST_ASSERT(p8est_quadrant_overlaps(&pquad, quad));
#endif // P4EST_ENABLE_DEBUG

  index += tree->quadrants_offset;

  P4EST_ASSERT(
      0 <= index &&
      index <
          forest_info.at(static_cast<int>(forest)).p4est->local_num_quadrants);

  return index;
}

p4est_locidx_t p4est_utils_pos_qid_ghost(forest_order forest,
                                         p8est_ghost_t *ghost,
                                         const double pos[3]) {
  p8est_tree_t *tree;
  p8est_quadrant_t q;
  p4est_utils_find_qid_prepare(forest, pos, &tree, &q);

  p4est_locidx_t index = p8est_find_lower_bound_overlap_piggy(
      &ghost->ghosts, &q, 0.5 * ghost->ghosts.elem_count);

#ifdef P4EST_ENABLE_DEBUG
  p8est_quadrant_t *quad = p4est_quadrant_array_index(&ghost->ghosts, index);
  P4EST_ASSERT(p8est_quadrant_overlaps(&q, quad));
#endif // P4EST_ENABLE_DEBUG

  P4EST_ASSERT(0 <= index && index < ghost->ghosts.elem_count);

  return index;
}

// CAUTION: Currently LB only
int coarsening_criteria(p8est_t *p8est, p4est_topidx_t which_tree,
                        p8est_quadrant_t **quads) {
  double pos[3];
  int inside = 1;
  for (int i = 0; i < P8EST_CHILDREN; ++i) {
    p4est_utils_get_front_lower_left(p8est, which_tree, quads[i], pos);
    // refine front lower left quadrant(s)
    std::array<double, 3> ref_min = {0.75, 0.00, 0.25};
    std::array<double, 3> ref_max = {1.00, 0.25, 0.50};
    if ((ref_min[0] <= pos[0] && pos[0] < ref_max[0]) &&
        (ref_min[1] <= pos[1] && pos[1] < ref_max[1]) &&
        (ref_min[2] <= pos[2] && pos[2] < ref_max[2])) {
      inside &= 1;
    } else {
      inside = 0;
    }
  }
  return inside;
}

int refinement_criteria(p8est_t *p8est, p4est_topidx_t which_tree,
                        p8est_quadrant_t *q) {
  double pos[3];
  p4est_utils_get_front_lower_left(p8est, which_tree, q, pos);
  // refine front lower left quadrant(s)
  std::array<double, 3> ref_min = {0.25, 0., 0.25};
  std::array<double, 3> ref_max = {0.5, 0.25, 0.5};
  if ((ref_min[0] <= pos[0] && pos[0] < ref_max[0]) &&
      (ref_min[1] <= pos[1] && pos[1] < ref_max[1]) &&
      (ref_min[2] <= pos[2] && pos[2] < ref_max[2])) {
    return 1;
  }
  return 0;
}

int p4est_utils_adapt_grid() {
  // 1st step: alter copied grid and map data between grids.
  const p4est_utils_forest_info_t &current_forest =
      p4est_utils_get_forest_info(forest_order::adaptive_LB);

  // FIXME remove DEBUG
  char filename_pre[42];
  snprintf(filename_pre, 42, "pre_gridchange_%i", (int)(sim_time / time_step));
  sc_array_t *vel;
  p4est_locidx_t num_cells = current_forest.p4est->local_num_quadrants;
  vel = sc_array_new_size(sizeof(double), P8EST_DIM * num_cells);

  lbadapt_get_velocity_values(vel);

  /* create VTK output context and set its parameters */
  p8est_vtk_context_t *context = p8est_vtk_context_new(lb_p8est, filename_pre);
  p8est_vtk_context_set_scale(context, 1); /* quadrant at almost full scale */

  /* begin writing the output files */
  context = p8est_vtk_write_header(context);
  SC_CHECK_ABORT(context != NULL,
                 P8EST_STRING "_vtk: Error writing vtk header");
  // clang-format off
  context = p8est_vtk_write_cell_dataf(context,
                                       1, /* write tree indices */
                                       1, /* write the refinement level */
                                       1, /* write the mpi process id */
                                       0, /* do not wrap the mpi rank */
                                       0, /* no custom cell scalar data */
                                       1, /* write velocity as vector cell
                                             data */
                                       "velocity", vel, context);
  // clang-format on
  SC_CHECK_ABORT(context != NULL, P8EST_STRING "_vtk: Error writing cell data");

  const int retval = p8est_vtk_write_footer(context);
  SC_CHECK_ABORT(!retval, P8EST_STRING "_vtk: Error writing footer");

  /* free memory */
  sc_array_destroy(vel);
  // FIXME remove DEBUG

  p8est_t *p4est_adapted = p8est_copy(current_forest.p4est, 0);
  p8est_refine_ext(p4est_adapted, 0, lbpar.max_refinement_level,
                   refinement_criteria, 0, 0);
  p8est_coarsen_ext(p4est_adapted, 0, 0, coarsening_criteria, 0, 0);
  p8est_balance_ext(p4est_adapted, P8EST_CONNECT_FULL, 0, 0);

  // only perform data mapping, communication, etc. if the forests have actually
  // changed
  if (!p8est_is_equal(lb_p8est, p4est_adapted, 0)) {
    // 0th step: de-allocate invalid storage and data-structures
    p4est_utils_deallocate_levelwise_storage(lbadapt_ghost_data);
    p8est_ghostvirt_destroy(lbadapt_ghost_virt);
    p8est_ghost_destroy(lbadapt_ghost);

    // 1st step: locally map data between forests.
    lbadapt_payload_t *mapped_data_flat =
        P4EST_ALLOC_ZERO(lbadapt_payload_t, p4est_adapted->local_num_quadrants);
    p4est_utils_post_gridadapt_map_data(lb_p8est, lbadapt_mesh, p4est_adapted,
                                        lbadapt_local_data, mapped_data_flat);
    // cleanup
    p4est_utils_deallocate_levelwise_storage(lbadapt_local_data);
    p8est_mesh_destroy(lbadapt_mesh);
    p8est_destroy(lb_p8est);

    // 2nd step: partition grid and transfer data to respective processors
    // FIXME: Interface to Steffen's partitioning logic
    p8est_t *p4est_partitioned = p8est_copy(p4est_adapted, 0);
    p8est_partition_ext(p4est_partitioned, 1, lbadapt_partition_weight);
    std::vector<std::vector<lbadapt_payload_t>> data_partitioned(
        p4est_partitioned->mpisize, std::vector<lbadapt_payload_t>());
    p4est_utils_post_gridadapt_data_partition_transfer(
        p4est_adapted, p4est_partitioned, mapped_data_flat, data_partitioned);
    // cleanup
    p8est_destroy(p4est_adapted);
    P4EST_FREE(mapped_data_flat);

    // 3rd step: Insert data into new levelwise data-structure and prepare next
    //           integration step
    lbadapt_ghost = p8est_ghost_new(p4est_partitioned, P8EST_CONNECT_FULL);
    lbadapt_mesh = p8est_mesh_new_ext(p4est_partitioned, lbadapt_ghost, 1, 1, 1,
                                      P8EST_CONNECT_FULL);
    lbadapt_ghost_virt =
        p8est_ghostvirt_new(p4est_partitioned, lbadapt_ghost, lbadapt_mesh);
    p4est_utils_allocate_levelwise_storage(lbadapt_local_data, lbadapt_mesh,
                                           true);
    p4est_utils_allocate_levelwise_storage(lbadapt_ghost_data, lbadapt_mesh,
                                           false);
    p4est_utils_post_gridadapt_insert_data(
        p4est_partitioned, lbadapt_mesh, data_partitioned, lbadapt_local_data);
    lb_p8est = p4est_partitioned;

    // synchronize ghost data for next collision step
    std::vector<lbadapt_payload_t *> local_pointer(P8EST_QMAXLEVEL);
    std::vector<lbadapt_payload_t *> ghost_pointer(P8EST_QMAXLEVEL);
    prepare_ghost_exchange(lbadapt_local_data, local_pointer,
                           lbadapt_ghost_data, ghost_pointer);

    for (int level = 0; level <= current_forest.finest_level_global; ++level) {
      p8est_ghostvirt_exchange_data(
          lb_p8est, lbadapt_ghost_virt, level, sizeof(lbadapt_payload_t),
          (void **)local_pointer.data(), (void **)ghost_pointer.data());
    }

    std::vector<p4est_t *> forests;
#ifdef DD_P4EST
    forests.push_back(dd.p4est);
#endif // DD_P4EST
    forests.push_back(lb_p8est);
    p4est_utils_prepare(forests);
#ifdef DD_P4EST
    p4est_utils_partition_multiple_forests(forest_order::short_range,
                                           forest_order::adaptive_LB);
#endif // DD_P4EST
  } else {
    p8est_destroy(p4est_adapted);
  }
  // FIXME remove DEBUG
  char filename_post[42];
  snprintf(filename_post, 42, "post_gridchange_%i",
           (int)(sim_time / time_step));
  num_cells = lb_p8est->local_num_quadrants;
  vel = sc_array_new_size(sizeof(double), P8EST_DIM * num_cells);

  lbadapt_get_velocity_values(vel);

  /* create VTK output context and set its parameters */
  p8est_vtk_context_t *context_post =
      p8est_vtk_context_new(lb_p8est, filename_post);
  p8est_vtk_context_set_scale(context_post, 1);

  /* begin writing the output files */
  context_post = p8est_vtk_write_header(context_post);
  SC_CHECK_ABORT(context_post != NULL,
                 P8EST_STRING "_vtk: Error writing vtk header");
  // clang-format off
  context_post = p8est_vtk_write_cell_dataf(context_post,
                                            1, /* write tree indices */
                                            1, /* write the refinement level */
                                            1, /* write the mpi process id */
                                            0, /* do not wrap the mpi rank */
                                            0, /* no custom cell scalar data */
                                            1, /* write velocity as vector cell
                                                  data */
                                            "velocity", vel, context_post);
  // clang-format on
  SC_CHECK_ABORT(context_post != NULL,
                 P8EST_STRING "_vtk: Error writing cell data");

  const int retval_post = p8est_vtk_write_footer(context_post);
  SC_CHECK_ABORT(!retval_post, P8EST_STRING "_vtk: Error writing footer");

  /* free memory */
  sc_array_destroy(vel);
  // FIXME remove DEBUG

  return 0;
}

template <typename T>
int p4est_utils_post_gridadapt_map_data(
    p8est_t *p4est_old, p8est_mesh_t *mesh_old, p8est_t *p8est_new,
    std::vector<std::vector<T>> &local_data_levelwise, T *mapped_data_flat) {
  // counters
  int tid_old = p4est_old->first_local_tree;
  int tid_new = p4est_new->first_local_tree;
  int qid_old = 0, qid_new = 0;
  int tqid_old = 0, tqid_new = 0;

  // trees
  p8est_tree_t *curr_tree_old =
      p8est_tree_array_index(p4est_old->trees, tid_old);
  p8est_tree_t *curr_tree_new =
      p8est_tree_array_index(p4est_new->trees, tid_new);
  // quadrants
  p8est_quadrant_t *curr_quad_old, *curr_quad_new;

  int level_old, sid_old;
  int level_new;
  while (qid_old < p4est_old->local_num_quadrants &&
         qid_new < p4est_new->local_num_quadrants) {
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
    sid_old = mesh_old->quad_qreal_offset[qid_old];

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
      // old cell has been refined
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
  P4EST_ASSERT(qid_old == p4est_old->local_num_quadrants);
  P4EST_ASSERT(qid_new == p4est_new->local_num_quadrants);

  return 0;
}

template <typename T>
int p4est_utils_post_gridadapt_data_partition_transfer(
    p8est_t *p4est_old, p8est_t *p4est_new, T *data_mapped,
    std::vector<std::vector<T>> &data_partitioned) {
  // simple consistency checks
  P4EST_ASSERT(p4est_old->mpirank == p4est_new->mpirank);
  P4EST_ASSERT(p4est_old->mpisize == p4est_new->mpisize);
  P4EST_ASSERT(p4est_old->global_num_quadrants ==
               p4est_new->global_num_quadrants);

  int rank = p4est_old->mpirank;
  int size = p4est_old->mpisize;
  int lb_old_local = p4est_old->global_first_quadrant[rank];
  int ub_old_local = p4est_old->global_first_quadrant[rank + 1];
  int lb_new_local = p4est_new->global_first_quadrant[rank];
  int ub_new_local = p4est_new->global_first_quadrant[rank + 1];
  int lb_old_remote = 0;
  int ub_old_remote = 0;
  int lb_new_remote = 0;
  int ub_new_remote = 0;
  int data_length = 0;
  int send_offset = 0;

  int mpiret;
  MPI_Request r;
  std::vector<MPI_Request> requests(2 * size, MPI_REQUEST_NULL);

  // determine from which processors we receive quadrants
  /** there are 5 cases to distinguish
   * 1. no quadrants of neighbor need to be received; neighbor rank < rank
   * 2. some quadrants of neighbor need to be received; neighbor rank < rank
   * 3. all quadrants of neighbor need to be received from neighbor
   * 4. some quadrants of neighbor need to be received; neighbor rank > rank
   * 5. no quadrants of neighbor need to be received; neighbor rank > rank
   */
  for (int p = 0; p < size; ++p) {
    lb_old_remote = ub_old_remote;
    ub_old_remote = p4est_old->global_first_quadrant[p + 1];

    // number of quadrants from which payload will be received
    data_length = std::max(0,
                           std::min(ub_old_remote, ub_new_local) -
                               std::max(lb_old_remote, lb_new_local));

    // allocate receive buffer and wait for messages
    data_partitioned[p].resize(data_length);
    r = requests[p];
    mpiret =
        MPI_Irecv((void *)data_partitioned[p].data(), data_length * sizeof(T),
                  MPI_BYTE, p, 0, p4est_new->mpicomm, &r);
    requests[p] = r;
    SC_CHECK_MPI(mpiret);
  }

  // send respective quadrants to other processors
  for (int p = 0; p < size; ++p) {
    lb_new_remote = ub_new_remote;
    ub_new_remote = p4est_new->global_first_quadrant[p + 1];

    data_length = std::max(0,
                           std::min(ub_old_local, ub_new_remote) -
                               std::max(lb_old_local, lb_new_remote));

    r = requests[size + p];
    mpiret =
        MPI_Isend((void *)(data_mapped + send_offset), data_length * sizeof(T),
                  MPI_BYTE, p, 0, p4est_new->mpicomm, &r);
    requests[size + p] = r;
    SC_CHECK_MPI(mpiret);
    send_offset += data_length;
  }

  /** Wait for communication to finish */
  mpiret = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  SC_CHECK_MPI(mpiret);

  return 0;
}

template <typename T>
int p4est_utils_post_gridadapt_insert_data(
    p8est_t *p4est_new, p8est_mesh_t *mesh_new,
    std::vector<std::vector<T>> &data_partitioned,
    std::vector<std::vector<T>> &data_levelwise) {
  int size = p4est_new->mpisize;
  // counters
  int tid = p4est_new->first_local_tree;
  int qid = 0;
  int tqid = 0;

  // trees
  p8est_tree_t *curr_tree = p8est_tree_array_index(p4est_new->trees, tid);
  // quadrants
  p8est_quadrant_t *curr_quad;

  int level, sid;

  for (int p = 0; p < size; ++p) {
    for (int q = 0; q < data_partitioned[p].size(); ++q) {
      // wrap multiple trees
      if (tqid == curr_tree->quadrants.elem_count) {
        ++tid;
        P4EST_ASSERT(tid < p4est_new->trees->elem_count);
        curr_tree = p8est_tree_array_index(p4est_new->trees, tid);
        tqid = 0;
      }
      curr_quad = p8est_quadrant_array_index(&curr_tree->quadrants, tqid);
      level = curr_quad->level;
      sid = mesh_new->quad_qreal_offset[qid];
      std::memcpy(&data_levelwise[level][sid], &data_partitioned[p][q],
                  sizeof(T));
      ++tqid;
      ++qid;
    }
  }

  // verify that all real quadrants have been processed
  P4EST_ASSERT(qid == mesh_new->local_num_quadrants);

  return 0;
}

void p4est_utils_partition_multiple_forests(forest_order reference,
                                            forest_order modify) {
  p8est_t *p4est_ref = forest_info.at(static_cast<int>(reference)).p4est;
  p8est_t *p4est_mod = forest_info.at(static_cast<int>(modify)).p4est;
  P4EST_ASSERT(p4est_ref->mpisize == p4est_mod->mpisize);
  P4EST_ASSERT(p4est_ref->mpirank == p4est_mod->mpirank);
  P4EST_ASSERT(p8est_connectivity_is_equivalent(p4est_ref->connectivity,
                                                p4est_mod->connectivity));

  std::vector<p4est_locidx_t> num_quad_per_proc(p4est_ref->mpisize, 0);
  std::vector<p4est_locidx_t> num_quad_per_proc_global(p4est_ref->mpisize, 0);

  int tid = p4est_mod->first_local_tree;
  int tqid = 0;
  // trees
  p8est_tree_t *curr_tree;
  // quadrants
  p8est_quadrant_t *curr_quad;

  if (0 < p4est_mod->local_num_quadrants) {
    curr_tree = p8est_tree_array_index(p4est_mod->trees, tid);
  }

  // Check for each of the quadrants of the given p4est, to which MD cell it
  // maps
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
    printf("%i : quadrants lost while partitioning\n", this_node);
    errexit();
  }

  CELL_TRACE(printf("%i : repartitioned LB %i\n", this_node,
                    num_quad_per_proc_global[this_node]));

  // Repartition with the computed distribution
  int shipped =
      p8est_partition_given(p4est_mod, num_quad_per_proc_global.data());
  P4EST_GLOBAL_PRODUCTIONF(
      "Done " P8EST_STRING "_partition shipped %lld quadrants %.3g%%\n",
      (long long)shipped, shipped * 100. / p4est_mod->global_num_quadrants);
}

int fct_coarsen_cb(p4est_t *p4est, p4est_topidx_t tree_idx,
                   p4est_quadrant_t *quad[]) {
  p4est_t *cmp = (p4est_t *)p4est->user_pointer;
  p4est_tree_t *tree = p4est_tree_array_index(cmp->trees, tree_idx);
  for (int i = 0; i < tree->quadrants.elem_count; ++i) {
    p4est_quadrant_t *q = p4est_quadrant_array_index(&tree->quadrants, i);
    if (p4est_quadrant_overlaps(q, quad[0]) && q->level >= quad[0]->level)
      return 0;
  }
  return 1;
}

p4est_t *p4est_utils_create_fct(p4est_t *t1, p4est_t *t2) {
  p4est_t *fct = p4est_copy(t2, 0);
  fct->user_pointer = (void *)t1;
  p4est_coarsen(fct, 1, fct_coarsen_cb, NULL);
  return fct;
}

#endif // defined (LB_ADAPTIVE) || defined (DD_P4EST)
