#include "p4est_utils.hpp"

#if (defined(LB_ADAPTIVE) || defined(DD_P4EST))

#include "debug.hpp"

#include <array>
#include <cstring>
#include <p8est_algorithms.h>
#include <vector>

int p4est_utils_pos_to_proc(p8est_t *p4est, double pos[3]) { return 0; }

int64_t p4est_utils_cell_morton_idx(int x, int y, int z) {
  // p4est_quadrant_t c;
  // c.x = x; c.y = y; c.z = z;
  /*if (x < 0 || x >= grid_size[0])
    runtimeErrorMsg() << x << "x" << y << "x" << z << " no valid cell";
  if (y < 0 || y >= grid_size[1])
    runtimeErrorMsg() << x << "x" << y << "x" << z << " no valid cell";
  if (z < 0 || z >= grid_size[2])
    runtimeErrorMsg() << x << "x" << y << "x" << z << " no valid cell";*/

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

  // c.level = P4EST_QMAXLEVEL;
  // return p4est_quadrant_linear_id(&c,P4EST_QMAXLEVEL);
}

int64_t p4est_utils_pos_morton_idx(p8est_t *p4est, double pos[3]) { return 0; }

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

void p4est_utils_partition_multiple_forests(p8est_t *p4est_ref,
                                            p8est_t *p4est_mod) {
  P4EST_ASSERT(p4est_ref->mpisize == p4est_mod->mpisize);
  P4EST_ASSERT(p4est_ref->mpirank == p4est_mod->mpirank);
  P4EST_ASSERT(p8est_connectivity_is_equal(p4est_ref->connectivity,
                                           p4est_mod->connectivity));

  p4est_locidx_t num_quad_per_proc[p4est_ref->mpisize];
  p4est_locidx_t num_quad_per_proc_global[p4est_ref->mpisize];
  for (int p = 0; p < p4est_mod->mpisize; ++p) {
    num_quad_per_proc[p] = 0;
    num_quad_per_proc_global[p] = 0;
  }

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
      int proc = p4est_utils_pos_to_proc(p4est_ref, xyz);
      ++num_quad_per_proc[proc];
    }
    ++tqid;
  }

  // Gather this information over all processes
  MPI_Allreduce(num_quad_per_proc, num_quad_per_proc_global, p4est_mod->mpisize,
                P4EST_MPI_LOCIDX, MPI_SUM, p4est_mod->mpicomm);

  p4est_locidx_t sum = 0;

  for (int i = 0; i < p4est_mod->mpisize; ++i)
    sum += num_quad_per_proc_global[i];
  if (sum < p4est_mod->global_num_quadrants) {
    printf("%i : quadrants lost while partitioning\n", this_node);
    return;
  }

  CELL_TRACE(printf("%i : repartitioned LB %i\n", this_node,
                    num_quad_per_proc_global[this_node]));

  // Repartition with the computed distribution
  int shipped = p8est_partition_given(p4est_mod, num_quad_per_proc_global);
  P4EST_GLOBAL_PRODUCTIONF(
      "Done " P8EST_STRING "_partition shipped %lld quadrants %.3g%%\n",
      (long long)shipped, shipped * 100. / p4est_mod->global_num_quadrants);
}

#endif // defined (LB_ADAPTIVE) || defined (DD_P4EST)
