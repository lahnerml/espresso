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

