#include "utils.hpp"

#include <cstring>

char *strcat_alloc(char *left, const char *right) {
  if (!left) {
    char *res = (char *)Utils::malloc(strlen(right) + 1);
    strcpy(res, right);
    return res;
  } else {
    char *res = (char *)Utils::realloc(left, strlen(left) + strlen(right) + 1);
    strcat(res, right);
    return res;
  }
}

template <typename T>
int post_gridadapt_map_data(p8est_t *p4est_old, p8est_mesh_t *mesh_old,
                            p8est_t *p4est_new, T **local_data_levelwise,
                            T *mapped_data_flat) {
  // counters
  int tid_old = 0, tid_new = 0;
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
      memcpy(&mapped_data_flat[qid_new],
             &local_data_levelwise[level_old][sid_old], sizeof(T));
      ++qid_old;
      ++qid_new;
      ++tqid_old;
      ++tqid_new;
    } else if (level_old + 1 == level_new) {
      // old cell has been coarsened
      for (int child = 0; child < P8EST_CHILDREN; ++child) {
        data_restriction(&local_data_levelwise[level_old][sid_old],
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
        data_interpolation(&local_data_levelwise[level_old][sid_old],
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
int post_gridadapt_data_partition_transfer(p8est_t *p4est_old,
                                           p8est_t *p4est_new, T *data_mapped,
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
int post_gridadapt_insert_data(p8est_t *p4est_new, p8est_mesh_t *mesh_new,
                               std::vector<T> **data_partitioned,
                               T **data_levelwise) {
  int size = p4est_new->mpisize;
  // counters
  int tid = 0;
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
