#include "p4est_utils.hpp"

#if (defined(LB_ADAPTIVE) || defined(DD_P4EST))

#include "debug.hpp"
#include "domain_decomposition.hpp"
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

const p4est_utils_forest_info_t &p4est_utils_get_forest_info(forest_order fo) {
  // Use at() here because forest_info might not have been initialized yet.
  return forest_info.at(static_cast<int>(fo));
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
    // set start index; if first tree is not completely owned by
    // current
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
      /* get local max level */
      if (insert_elem.finest_level_local < tree->maxlevel) {
        insert_elem.finest_level_local = tree->maxlevel;
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
  insert_elem.finest_level_ghost = insert_elem.finest_level_global;

  // ensure monotony
  P4EST_ASSERT(std::is_sorted(insert_elem.tree_quadrant_offset_synced.begin(),
                              insert_elem.tree_quadrant_offset_synced.end()));

  for (int i = 0; i < p4est->mpisize; ++i) {
    p4est_quadrant_t *q = &p4est->global_first_position[i];
    double xyz[3];
    p4est_qcoord_to_vertex(p4est->connectivity, q->p.which_tree, q->x, q->y,
                           q->z, xyz);

    insert_elem.first_quad_morton_idx[i] = p4est_utils_pos_morton_idx_global(
        p4est, insert_elem.finest_level_global,
        insert_elem.tree_quadrant_offset_synced, xyz);
  }
  insert_elem.first_quad_morton_idx[p4est->mpisize] =
      p4est->global_num_quadrants;
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
  p4est_utils_forest_info_t current_forest =
      forest_info.at(static_cast<int>(forest));
  p8est_t *p4est = current_forest.p4est;
  int qid = p4est_utils_pos_morton_idx_global(forest, pos);

  int p = std::distance(current_forest.first_quad_morton_idx.begin(),
                        std::upper_bound(
                            current_forest.first_quad_morton_idx.begin(),
                            current_forest.first_quad_morton_idx.end(), qid)) -
          1;

  P4EST_ASSERT(0 <= p && p < p4est->mpisize);

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

#ifndef LB_ADAPTIVE
      // As pure MD allows for box_l != 1.0, "pos" will be in [0,box_l) and
      // not in [0,1). So manually scale the trees to fill [0,box_l).
      c[ci][0] *= box_l[0] / dd_p4est_num_trees_in_dir(0);
      c[ci][1] *= box_l[1] / dd_p4est_num_trees_in_dir(1);
      c[ci][2] *= box_l[2] / dd_p4est_num_trees_in_dir(2);
#endif
    }

    // find lower left and upper right corner of forest
    std::array<double, 3> pos_min = {0., 0., 0.};
    std::array<double, 3> pos_max = {box_l[0], box_l[1], box_l[2]};
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

  double spos[3] = { pos[0], pos[1], pos[2] };
#ifndef LB_ADAPTIVE
  // In case of pure MD arbitrary numbers are allowed for box_l.
  // Scale "spos" such that it corresponds to a box_l of 1.0
  for (int i = 0; i < 3; ++i) {
    spos[i] /= box_l[i] / dd_p4est_num_trees_in_dir(i);
  }
#endif // !LB_ADAPTIVE

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
  p4est_utils_forest_info_t &current_p4est =
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
  p4est_utils_forest_info_t &current_p4est =
      forest_info.at(static_cast<int>(forest));
  p8est_t *p4est = current_p4est.p4est;

  // find correct tree
  int tid = p4est_utils_map_pos_to_tree(p4est, pos);
  int level = current_p4est.finest_level_global;
  *tree = p4est_tree_array_index(p4est->trees, tid);

  double first_pos[3];
  int shifted_pos[3];
  p4est_qcoord_to_vertex(p4est->connectivity, tid, 0, 0, 0, first_pos);

  int64_t pidx;
  for (int i = 0; i < P8EST_DIM; ++i) {
#ifdef LB_ADAPTIVE
    // Trees have a base length of 1
    shifted_pos[i] = (pos[i] - first_pos[i]) * (1 << level);
#else
    // Trees might not have a base length of 1
    shifted_pos[i] =
        (pos[i] * dd_p4est_num_trees_in_dir(i) / box_l[i] - first_pos[i]) *
        (1 << level);
#endif
  }
  pidx = p4est_utils_cell_morton_idx(shifted_pos[0], shifted_pos[1],
                                     shifted_pos[2]);
  p4est_quadrant_set_morton(pquad, level, pidx);
  pquad->p.which_tree = tid;

  return 0;
}

p4est_locidx_t p4est_utils_pos_qid_local(forest_order forest,
                                         const double pos[3]) {
  p4est_utils_forest_info_t &current_p4est =
      forest_info.at(static_cast<int>(forest));
  p8est_t *p4est = current_p4est.p4est;

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

template <typename T>
int p4est_utils_post_gridadapt_map_data(p8est_t *p4est_old,
                                        p8est_mesh_t *mesh_old,
                                        p8est_t *p4est_new,
                                        T **local_data_levelwise,
                                        T *mapped_data_flat) {
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
    } else if (level_old + 1 == level_new) {
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
    } else if (level_old == level_new + 1) {
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
  return 0;
}

template <typename T>
int p4est_utils_post_gridadapt_data_partition_transfer(
    p8est_t *p4est_old, p8est_t *p4est_new, T *data_mapped,
    std::vector<T> **data_partitioned) {
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
  sc_MPI_Request *r;
  sc_array_t *requests;
  requests = sc_array_new(sizeof(sc_MPI_Request));

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

    data_length = std::max(0,
                           std::min(ub_old_remote, ub_new_local) -
                               std::max(lb_old_remote, ub_new_local));

    // allocate receive buffer and wait for messages
    data_partitioned[p] = new std::vector<T>(data_length);
    r = (sc_MPI_Request *)sc_array_push(requests);
    mpiret = sc_MPI_Irecv((void *)data_partitioned[p]->begin(),
                          data_length * sizeof(T), sc_MPI_BYTE, p, 0,
                          p4est_new->mpicomm, r);
    SC_CHECK_MPI(mpiret);
  }

  // send respective quadrants to other processors
  for (int p = 0; p < size; ++p) {
    lb_new_remote = ub_new_remote;
    ub_new_remote = p4est_new->global_first_quadrant[p + 1];

    data_length = std::max(0,
                           std::min(ub_old_local, ub_new_remote) -
                               std::max(lb_old_local, ub_new_remote));

    r = (sc_MPI_Request *)sc_array_push(requests);
    mpiret = sc_MPI_Isend((void *)(data_mapped + send_offset * sizeof(T)),
                          data_length * sizeof(T), sc_MPI_BYTE, p, 0,
                          p4est_new->mpicomm, r);
    SC_CHECK_MPI(mpiret);
    send_offset += data_length;
  }

  /** Wait for communication to finish */
  mpiret =
      sc_MPI_Waitall(requests->elem_count, (sc_MPI_Request *)requests->array,
                     sc_MPI_STATUSES_IGNORE);
  SC_CHECK_MPI(mpiret);
  sc_array_destroy(requests);

  return 0;
}

template <typename T>
int p4est_utils_post_gridadapt_insert_data(p8est_t *p4est_new,
                                           p8est_mesh_t *mesh_new,
                                           std::vector<T> **data_partitioned,
                                           T **data_levelwise) {
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
    for (int q = 0; q < data_partitioned[p]->size(); ++q) {
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
      std::memcpy(&data_levelwise[level][sid], &data_partitioned[p]->at(q),
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
