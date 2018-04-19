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
#include <fstream>
#include <iterator>
#include <limits>
#include <mpi.h>
#include <p8est_algorithms.h>
#include <p8est_bits.h>
#include <p8est_search.h>
#include <vector>

// For intrinsics version of cell_morton_idx
#ifdef __BMI2__
#include <x86intrin.h>
#endif

static std::vector<p4est_utils_forest_info_t> forest_info;

p4est_parameters p4est_params = {
  // min_ref_level
  -1,
  // max_ref_level
  -1,
  // threshold_velocity
  {0.0, 1.0},
  // threshold_vorticity
  {0.0, 1.0},
};

#ifdef LB_ADAPTIVE
int lb_conn_brick[3] = {0, 0, 0};
#endif // LB_ADAPTIVE

double coords_for_regional_refinement[6] = {
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max()};

// CAUTION: Do ONLY use this pointer in p4est_utils_perform_adaptivity_step
std::vector<int> *flags;

const p4est_utils_forest_info_t &p4est_utils_get_forest_info(forest_order fo) {
  // Use at() here because forest_info might not have been initialized yet.
  return forest_info.at(static_cast<int>(fo));
}

static p4est_utils_forest_info_t p4est_to_forest_info(p4est_t *p4est) {
  if (p4est) {
    // fill element to insert
    p4est_utils_forest_info_t insert_elem(p4est);

    // allocate a local send buffer to insert local quadrant offsets
    std::vector<p4est_locidx_t> local_tree_offsets(p4est->trees->elem_count);

    // only inspect local trees if current process hosts quadrants
    if (p4est->local_num_quadrants != 0) {
      for (int i = p4est->first_local_tree; i <= p4est->last_local_tree; ++i) {
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
      if (i < n_nodes) {
        double xyz[3];
        insert_elem.p4est_space_idx[i] =
            p4est_utils_global_idx(q, q->p.which_tree);
      } else {
        double n_trees[3];
#if defined(LB_ADAPTIVE)
        for (int i = 0; i < P8EST_DIM; ++i)
          n_trees[i] = lb_conn_brick[i];
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
    return nullptr;
  }
}

void p4est_utils_prepare(std::vector<p8est_t *> p4ests) {
  forest_info.clear();

  std::transform(std::begin(p4ests), std::end(p4ests),
                 std::back_inserter(forest_info), p4est_to_forest_info);
}

void p4est_utils_rebuild_p4est_structs(p4est_connect_type_t btype) {
  std::vector<p4est_t *> forests;
#ifdef DD_P4EST
  forests.push_back(dd_p4est_get_p4est());
#endif // DD_P4EST
#ifdef LB_ADAPTIVE
  forests.push_back(adapt_p4est);
#endif // LB_ADAPTIVE
  p4est_utils_prepare(forests);
#if defined(DD_P4EST) && defined(LB_ADAPTIVE)
  p4est_utils_partition_multiple_forests(forest_order::short_range,
                                         forest_order::adaptive_LB);
#elif defined(DD_P4EST)
  p4est_partition(dd_p4est_get_p4est(), 1, nullptr);
#elif defined(LB_ADAPTIVE)
  p4est_partition(adapt_p4est, 1, lbadapt_partition_weight);
#endif // DD_P4EST
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

#ifdef __BMI2__
static unsigned
_d2x(unsigned d, unsigned mask)
{
  return _pext_u32(d, mask);
}
#endif

std::array<unsigned int, 3> p4est_utils_idx_to_pos (int64_t idx) {
#ifdef __BMI2__
  static const unsigned mask_x = 0x49249249;
  static const unsigned mask_y = 0x92492492;
  static const unsigned mask_z = 0x24924924;

  std::array<unsigned int, 3> res = {{
      _d2x(idx, mask_x),
      _d2x(idx, mask_y),
      _d2x(idx, mask_z)
  }};
#else
  unsigned int x = 0;
  unsigned int y = 0;
  unsigned int z = 0;
  for (int i = 0; i < 21; ++i) {
    x |= (idx & (1 << (3 * i + 0))) << i;
    y |= (idx & (1 << (3 * i + 1))) << i;
    z |= (idx & (1 << (3 * i + 2))) << i;
  }
  std::array<unsigned int, 3> res = {{x, y, z}};
#endif
  return res;
}

// Returns the morton index for given cartesian coordinates.
// Note: This is not the index of the p4est quadrants. But the ordering is the same.
int64_t p4est_utils_cell_morton_idx(int x, int y, int z) {
#ifdef __BMI2__
  //#warning "Using BMI2 for cell_morton_idx"
  static const unsigned mask_x = 0x49249249;
  static const unsigned mask_y = 0x92492492;
  static const unsigned mask_z = 0x24924924;

  return _pdep_u32(x, mask_x)
           | _pdep_u32(y, mask_y)
           | _pdep_u32(z, mask_z);
#else
//#warning "BMI2 not detected: Using slow loop version for cell_morton_idx"
  int64_t idx = 0;
  int64_t pos = 1;

  for (int i = 0; i < 21; ++i) {
    if ((x&1)) idx += pos;
    x >>= 1; pos <<= 1;
    if ((y&1)) idx += pos;
    y >>= 1; pos <<= 1;
    if ((z&1)) idx += pos;
    z >>= 1; pos <<= 1;
  }

  return idx;
#endif
}

int64_t p4est_utils_global_idx(p8est_quadrant_t *q, p4est_topidx_t tree) {
  int x, y, z;
  double xyz[3];
  p8est_qcoord_to_vertex(adapt_p4est->connectivity, tree, q->x, q->y, q->z, xyz);
  x = xyz[0] * (1 << p4est_params.max_ref_level);
  y = xyz[1] * (1 << p4est_params.max_ref_level);
  z = xyz[2] * (1 << p4est_params.max_ref_level);

  return p4est_utils_cell_morton_idx(x, y, z);
}

int64_t p4est_utils_pos_to_index(forest_order forest, double xyz[3]) {
  const auto &fi = forest_info.at(static_cast<int>(forest));
  std::array<double, 3> pos = boxl_to_treecoords_copy(xyz);
  int xyz_mod[3];
  for (int i = 0; i < P8EST_DIM; ++i)
    xyz_mod[i] = pos[i] * (1 << fi.finest_level_global);
  return p4est_utils_cell_morton_idx(xyz_mod[0], xyz_mod[1], xyz_mod[2]);
}

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
  p4est_gloidx_t cmp_idx;
  p4est_locidx_t count, step, index, first;
  p4est_quadrant_t *q;
  first = 0;
  count = adapt_p4est->local_num_quadrants;
  while (0 < count) {
    step = 0.5 * count;
    index = first + step;
    q = p4est_mesh_get_quadrant(adapt_p4est, adapt_mesh, index);
    p4est_topidx_t tid = adapt_mesh->quad_to_tree[index];
    cmp_idx = p4est_utils_global_idx(q, tid);
    if (cmp_idx < idx) {
      // if we found something smaller: move to latter part of search space
      first = index + 1;
      count -= step + 1;
    } else if (cmp_idx == idx) {
      return index;
    }
    else {
      // else limit search space to half the array
      count = step;
    }
  }
  return first;
}

p4est_locidx_t p4est_utils_pos_to_qid(forest_order forest, double *xyz) {
  return p4est_utils_idx_to_qid(forest, p4est_utils_pos_to_index(forest, xyz));
}

p4est_locidx_t p4est_utils_idx_to_qid(forest_order forest, p4est_gloidx_t idx) {
  P4EST_ASSERT(forest == forest_order::adaptive_LB);
  return bin_search_loc_quads(idx);
}

// Find the process that handles the position
int p4est_utils_pos_to_proc(forest_order forest, double* xyz) {
  return p4est_utils_idx_to_proc(forest, p4est_utils_pos_to_index(forest, xyz));
}

int p4est_utils_idx_to_proc(forest_order forest, p4est_gloidx_t idx) {
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
               ((*flags)[qid + i] == 2);
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
      ((1 == (*flags)[qid] || refine))) {
    return 1;
  }
#endif // LB_ADAPTIVE
  return 0;
}

int p4est_utils_collect_flags(std::vector<int> *flags) {
#ifdef LB_ADAPTIVE
  // get refinement string for first grid change operation
  // velocity
  // Euclidean norm
  castable_unique_ptr<sc_array_t> vel_values =
      sc_array_new_size(sizeof(double), 3 * adapt_p4est->local_num_quadrants);
  lbadapt_get_velocity_values(vel_values);
  double v;
  double v_min = std::numeric_limits<double>::max();
  double v_max = std::numeric_limits<double>::min();
  for (int qid = 0; qid < adapt_p4est->local_num_quadrants; ++qid) {
    v = sqrt(Utils::sqr(*(double *)sc_array_index(vel_values, 3 * qid)) +
             Utils::sqr(*(double *)sc_array_index(vel_values, 3 * qid + 1)) +
             Utils::sqr(*(double *)sc_array_index(vel_values, 3 * qid + 2)));
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

  // vorticity
  // max norm
  castable_unique_ptr<sc_array_t> vort_values =
      sc_array_new_size(sizeof(double), 3 * adapt_p4est->local_num_quadrants);
  lbadapt_get_vorticity_values(vort_values);
  double vort_min = std::numeric_limits<double>::max();
  double vort_max = std::numeric_limits<double>::min();
  double vort_temp;
  for (int qid = 0; qid < adapt_p4est->local_num_quadrants; ++qid) {
    for (int d = 0; d < P4EST_DIM; ++d) {
      vort_temp = abs(*(double*) sc_array_index(vort_values, 3 * qid + d));
      if (vort_temp < vort_min) {
        vort_min = vort_temp;
      }
      if (vort_temp > vort_max) {
        vort_max = vort_temp;
      }
    }
  }
  // sync
  vort_temp  = vort_min;
  MPI_Allreduce(&vort_temp, &vort_min, 1, MPI_DOUBLE, MPI_MIN,
                adapt_p4est->mpicomm);
  vort_temp  = vort_max;
  MPI_Allreduce(&vort_temp, &vort_max, 1, MPI_DOUBLE, MPI_MAX,
                adapt_p4est->mpicomm);

  // traverse forest and decide if the current quadrant is to be refined or
  // coarsened
  for (int qid = 0; qid < adapt_p4est->local_num_quadrants; ++qid) {
    // velocity
    double v = sqrt(Utils::sqr(*(double*) sc_array_index(vel_values, 3 * qid)) +
                    Utils::sqr(*(double*) sc_array_index(vel_values, 3 * qid + 1)) +
                    Utils::sqr(*(double*) sc_array_index(vel_values, 3 * qid + 2)));
    // Note, that this formulation stems from the fact that velocity is 0 at
    // boundaries
    if (p4est_params.threshold_velocity[1] * (v_max - v_min) < (v - v_min)) {
      (*flags)[qid] = 1;
    }
    else if ((1 != (*flags)[qid]) &&
             (v - v_min < p4est_params.threshold_velocity[0] *
                          (v_max - v_min))) {
      (*flags)[qid] = 2;
    }

    // vorticity
    double vort = std::numeric_limits<double>::min();
    for (int d = 0; d < P4EST_DIM; ++d) {
      vort_temp = abs(*(double*) sc_array_index(vort_values, 3 * qid + d));
      if (vort < vort_temp) {
        vort = vort_temp;
      }
    }
    if (p4est_params.threshold_vorticity[1] * (vort_max - vort_min) <
        (vort - vort_min)) {
      (*flags)[qid] = 1;
    }
    else if ((1 != (*flags)[qid]) &&
             (vort - vort_min < p4est_params.threshold_vorticity[0] *
                                (vort_max - vort_min))) {
      (*flags)[qid] = 2;
    }
  }
#endif // LB_ADAPTIVE
  return 0;
}

/** Dummy initialization function for quadrants created in refinement step
 */
void p4est_utils_qid_dummy (p8est_t *p8est, p4est_topidx_t which_tree,
                            p8est_quadrant_t *q) {
  q->p.user_long = -1;
}

int p4est_utils_perform_adaptivity_step() {
#ifdef LB_ADAPTIVE
  p4est_connect_type_t btype = P4EST_CONNECT_FULL;

  // 1st step: alter copied grid and map data between grids.
  // collect refinement and coarsening flags.
  flags = new std::vector<int>(adapt_p4est->local_num_quadrants, 0);
  p4est_utils_collect_flags(flags);

  p4est_iterate(adapt_p4est, adapt_ghost, nullptr, lbadapt_init_qid_payload,
                nullptr, nullptr, nullptr);

  // copy forest and perform refinement step.
  p8est_t *p4est_adapted = p8est_copy(adapt_p4est, 0);
  P4EST_ASSERT(p4est_is_equal(p4est_adapted, adapt_p4est, 0));
  p8est_refine_ext(p4est_adapted, 0, p4est_params.max_ref_level,
                   refinement_criteria, p4est_utils_qid_dummy, nullptr);
  // perform coarsening step
  p8est_coarsen_ext(p4est_adapted, 0, 0, coarsening_criteria, nullptr, nullptr);
  delete flags;
  // balance forest after grid change
  p8est_balance_ext(p4est_adapted, P8EST_CONNECT_FULL, nullptr, nullptr);

  // 2nd step: locally map data between forests.
  // de-allocate invalid storage and data-structures
  p4est_utils_deallocate_levelwise_storage(lbadapt_ghost_data);
  // locally map data between forests.
  lbadapt_payload_t *mapped_data_flat =
      P4EST_ALLOC_ZERO(lbadapt_payload_t, p4est_adapted->local_num_quadrants);
  p4est_utils_post_gridadapt_map_data(adapt_p4est, adapt_mesh, adapt_virtual,
                                      p4est_adapted, lbadapt_local_data,
                                      mapped_data_flat);
  // cleanup
  p4est_utils_deallocate_levelwise_storage(lbadapt_local_data);
  adapt_virtual.reset();
  adapt_mesh.reset();
  adapt_p4est.reset();

  // 3rd step: partition grid and transfer data to respective new owner ranks
  // FIXME: Interface to Steffen's partitioning logic
  // FIXME: Synchronize partitioning between short-range MD and adaptive
  //        p4ests
  p8est_t *p4est_partitioned = p8est_copy(p4est_adapted, 0);
  p8est_partition_ext(p4est_partitioned, 1, lbadapt_partition_weight);
  std::vector<std::vector<lbadapt_payload_t>> data_partitioned(
      p4est_partitioned->mpisize, std::vector<lbadapt_payload_t>());
  p4est_utils_post_gridadapt_data_partition_transfer(
      p4est_adapted, p4est_partitioned, mapped_data_flat, data_partitioned);

  p4est_destroy(p4est_adapted);
  P4EST_FREE(mapped_data_flat);

  // 4th step: Insert received data into new levelwise data-structure
  adapt_p4est.reset(p4est_partitioned);
  adapt_ghost.reset(p4est_ghost_new(adapt_p4est, btype));
  adapt_mesh.reset(p4est_mesh_new_ext(adapt_p4est, adapt_ghost, 1, 1, 1,
                                      btype));
  adapt_virtual.reset(p4est_virtual_new_ext(adapt_p4est, adapt_ghost,
                                            adapt_mesh, btype, 1));
  adapt_virtual_ghost.reset(p4est_virtual_ghost_new(adapt_p4est, adapt_ghost,
                                                    adapt_mesh, adapt_virtual,
                                                    btype));
  p4est_utils_allocate_levelwise_storage(lbadapt_local_data, adapt_mesh,
                                         adapt_virtual, true);
  p4est_utils_allocate_levelwise_storage(lbadapt_ghost_data, adapt_mesh,
                                         adapt_virtual, false);
  p4est_utils_post_gridadapt_insert_data(p4est_partitioned, adapt_mesh,
                                         adapt_virtual, data_partitioned,
                                         lbadapt_local_data);

  // 5th step: Prepare next integration step
  std::vector<p4est_t *> forests;
#ifdef DD_P4EST
  forests.push_back(dd_p4est_get_p4est());
#endif // DD_P4EST
  forests.push_back(adapt_p4est);
  p4est_utils_prepare(forests);
  const p4est_utils_forest_info_t new_forest =
      p4est_utils_get_forest_info(forest_order::adaptive_LB);
  // synchronize ghost data for next collision step
  std::vector<lbadapt_payload_t *> local_pointer(P8EST_QMAXLEVEL);
  std::vector<lbadapt_payload_t *> ghost_pointer(P8EST_QMAXLEVEL);
  prepare_ghost_exchange(lbadapt_local_data, local_pointer,
                         lbadapt_ghost_data, ghost_pointer);
  for (int level = new_forest.coarsest_level_global;
       level <= new_forest.finest_level_global; ++level) {
    p4est_virtual_ghost_exchange_data_level (adapt_p4est, adapt_ghost,
                                             adapt_mesh, adapt_virtual,
                                             adapt_virtual_ghost, level,
                                             sizeof(lbadapt_payload_t),
                                             (void**)local_pointer.data(),
                                             (void**)ghost_pointer.data());
  }

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
    p8est_t *p4est_new, p8est_mesh_t *mesh_new, p4est_virtual_t *virtual_quads,
    std::vector<std::vector<T>> &data_partitioned,
    std::vector<std::vector<T>> &data_levelwise) {
  int size = p4est_new->mpisize;
  // counters
  unsigned int tid = p4est_new->first_local_tree;
  int qid = 0;
  unsigned int tqid = 0;

  // trees
  p8est_tree_t *curr_tree = p8est_tree_array_index(p4est_new->trees, tid);
  // quadrants
  p8est_quadrant_t *curr_quad;

  int level, sid;

  for (int p = 0; p < size; ++p) {
    for (unsigned int q = 0; q < data_partitioned[p].size(); ++q) {
      // wrap multiple trees
      if (tqid == curr_tree->quadrants.elem_count) {
        ++tid;
        P4EST_ASSERT(tid < p4est_new->trees->elem_count);
        curr_tree = p8est_tree_array_index(p4est_new->trees, tid);
        tqid = 0;
      }
      curr_quad = p8est_quadrant_array_index(&curr_tree->quadrants, tqid);
      level = curr_quad->level;
      sid = virtual_quads->quad_qreal_offset[qid];
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
        (long long) shipped, shipped * 100. / p4est_mod->global_num_quadrants);
  }
  else {
    if (this_node == 0) {
      std::cerr
          << "Not all p4ests have been created yet. This may happen during"
          << " initialization." << std::endl;
    }
  }
#endif // defined(DD_P4EST) && defined(LB_ADAPTIVE)
}

int fct_coarsen_cb(p4est_t *p4est, p4est_topidx_t tree_idx,
                   p4est_quadrant_t *quad[]) {
  p4est_t *cmp = (p4est_t *)p4est->user_pointer;
  p4est_tree_t *tree = p4est_tree_array_index(cmp->trees, tree_idx);
  for (unsigned int i = 0; i < tree->quadrants.elem_count; ++i) {
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

bool p4est_utils_check_alignment(const p4est_t *t1, const p4est_t *t2) {
  if (!p4est_connectivity_is_equivalent(t1->connectivity, t2->connectivity)) return false;
  if (t1->first_local_tree != t2->first_local_tree) return false;
  if (t1->last_local_tree != t2->last_local_tree) return false;
  p4est_quadrant_t *q1 = &t1->global_first_position[t1->mpirank];
  p4est_quadrant_t *q2 = &t2->global_first_position[t2->mpirank];
  if (q1->x != q2->x && q1->y != q2->y && q1->z != q2->z) return false;
  q1 = &t1->global_first_position[t1->mpirank+1];
  q2 = &t2->global_first_position[t2->mpirank+1];
  if (q1->x != q2->x && q1->y != q2->y && q1->z != q2->z) return false;
  return true;
}

void p4est_utils_weighted_partition(p4est_t *t1, const std::vector<double> &w1,
                                    double a1, p4est_t *t2,
                                    const std::vector<double> &w2, double a2) {
  P4EST_ASSERT(p4est_utils_check_alignment(t1, t2));

  std::unique_ptr<p4est_t> fct(p4est_utils_create_fct(t1, t2));
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
    p4est_quadrant_t *q2 = p4est_quadrant_array_index(&t_t1->quadrants, q_id2);
    for (size_t q_idx = 0; q_idx < t_fct->quadrants.elem_count; ++q_idx) {
      p4est_quadrant_t *q_fct = p4est_quadrant_array_index(&t_fct->quadrants, q_idx);
      while (p4est_quadrant_overlaps(q_fct, q1)) {
        w_fct[w_idx] += a1*w1[w_id1++];
        ++t1_quads_per_fct_quad[w_idx];
        if (++q_id1 >= t_t1->quadrants.elem_count) {
          // complain if last quad in t1 does not overlap with last quad of FCT
          P4EST_ASSERT(q_idx == t_fct->quadrants.elem_count - 1);
          break;
        }
        q1 = p4est_quadrant_array_index(&t_t1->quadrants, q_id1);
      }
      while (p4est_quadrant_overlaps(q_fct, q2)) {
        w_fct[w_idx] += a2*w2[w_id2++];
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
    int proc = std::min<int>(w_fct[idx] / target, fct->mpisize - 1);
    t1_quads_per_proc[proc] += t1_quads_per_fct_quad[idx];
    t2_quads_per_proc[proc] += t2_quads_per_fct_quad[idx];
  }

  MPI_Allreduce(MPI_IN_PLACE, t1_quads_per_proc.data(), fct->mpisize,
                P4EST_MPI_LOCIDX, MPI_SUM, comm_cart);
  MPI_Allreduce(MPI_IN_PLACE, t2_quads_per_proc.data(), fct->mpisize,
                P4EST_MPI_LOCIDX, MPI_SUM, comm_cart);

  p4est_partition_given(t1, t1_quads_per_proc.data());
  p4est_partition_given(t2, t2_quads_per_proc.data());
}

#endif // defined (LB_ADAPTIVE) || defined (DD_P4EST)
