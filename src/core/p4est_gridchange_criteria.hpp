#ifndef P4EST_GRIDCHANGE_CRITERIA_HPP
#define P4EST_GRIDCHANGE_CRITERIA_HPP

#include "integrate.hpp"
#include "p4est_utils.hpp"

#include <array>

/** change grid based on a quadrant's position
 */
inline static int random_geometric(p8est_t *p8est, p4est_topidx_t which_tree,
                                   p8est_quadrant_t *q,
                                   std::array<double, 3> ref_min,
                                   std::array<double, 3> ref_max) {
  P4EST_ASSERT((int)((sim_time / time_step) + 0.5) % steps_until_grid_change == 0);
  double pos[3];
  p4est_utils_get_front_lower_left(p8est, which_tree, q, pos);
  if ((ref_min[0] <= pos[0] && pos[0] < ref_max[0]) &&
      (ref_min[1] <= pos[1] && pos[1] < ref_max[1]) &&
      (ref_min[2] <= pos[2] && pos[2] < ref_max[2])) {
    return 1;
  }
  return 0;
}

/** change grid based on a quadrant's level
 */
inline static int mirror_refinement_pattern(p8est_t *p8est,
                                            p4est_topidx_t which_tree,
                                            p8est_quadrant_t *q) {
  P4EST_ASSERT((int)((sim_time / time_step) + 0.5) % steps_until_grid_change == 0);
  double pos[3];
  p4est_utils_get_front_lower_left(p8est, which_tree, q, pos);

  bool change_grid = false;
  // array containing areas that need to be changed if cell is contained there
  double ref_areas[4][3][2] = {{{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}},
                               {{0.5, 1.0}, {0.5, 1.0}, {0.0, 0.5}},
                               {{0.5, 1.0}, {0.0, 0.5}, {0.5, 1.0}},
                               {{0.0, 0.5}, {0.5, 1.0}, {0.5, 1.0}}};
  for (int i = 0; i < P8EST_HALF; ++i) {
    // if
    if ((ref_areas[i][0][0] <= pos[0] && pos[0] < ref_areas[i][0][1]) &&
        (ref_areas[i][1][0] <= pos[1] && pos[1] < ref_areas[i][1][1]) &&
        (ref_areas[i][2][0] <= pos[2] && pos[2] < ref_areas[i][2][1])) {
      change_grid = true;
    }
    if (change_grid)
      break;
  }
  if ((int)(sim_time / time_step) % (2 * steps_until_grid_change) == 0) {
    change_grid = !change_grid;
  }
  return change_grid;
}

#endif // P4EST_GRIDCHANGE_CRITERIA_HPP
