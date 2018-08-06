#ifndef ESPRESSO_COUPLING_HELPER_HPP
#define ESPRESSO_COUPLING_HELPER_HPP

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <vector>

template <typename T1, typename T2>
int get_linear_index(T1 a, T1 b, T1 c, T2 adim[3]) {
  return (a + adim[0] * (b + adim[1] * c));
}

typedef struct coupling_helper {
  int particle_id;
  std::array<double, 3> particle_position;
  std::array<double, 3> particle_force;
  std::vector<double> delta;
#ifdef LB_ADAPTIVE
  bool cell_positions_wrapped = true;
#else
  bool cell_positions_wrapped = false;
#endif
  std::vector<std::array <uint64_t, 3> > cell_positions;
  std::vector<std::array <uint64_t, 3> > wrapped_cell_positions;
  std::vector<std::array <double, 3> > fluid_force;

  void reset() {
    particle_id = -1;
    particle_position = {{-1., -1., -1.}};
    delta.clear();
#ifdef LB_ADAPTIVE
    cell_positions_wrapped = true;
#else
    cell_positions_wrapped = false;
#endif
    cell_positions.clear();
    fluid_force.clear();
    particle_force = {{-1., -1., -1.}};
  }

  void preprocess_position() {
    if (!cell_positions_wrapped) {
      std::array<uint64_t , 3> wrapped_pos;
      for (int i = 0; i < cell_positions.size(); ++i) {
        wrapped_pos = {{0L, 0L, 0L}};
        for (int j = 0; j < 3; ++j) {
          if (cell_positions[i][j] == 0) {
            wrapped_pos[j] = 31;
          } else if (cell_positions[i][j] == 33) {
            wrapped_pos[j] = 0;
          } else {
            wrapped_pos[j] = cell_positions[i][j] - 1;
          }
        }
        wrapped_cell_positions.push_back(wrapped_pos);
      }
      cell_positions_wrapped = true;
    } else if(wrapped_cell_positions.empty()) {
      wrapped_cell_positions = cell_positions;
    }
  }

  std::string print(std::vector<uint64_t> &coupling_order) {
    preprocess_position();
    std::iota(coupling_order.begin(), coupling_order.end(), 0);
    std::sort(coupling_order.begin(), coupling_order.end(),
              [&](const uint64_t a, const uint64_t b) -> bool {
#ifdef LB_ADAPTIVE
              int n_q[3] = {32, 32, 32};
              return (get_linear_index(
                          cell_positions[a][0], cell_positions[a][1],
                          cell_positions[a][2], n_q) <
                      get_linear_index(
                          cell_positions[b][0], cell_positions[b][1],
                          cell_positions[b][2], n_q));
#else
              int n_q[3] = {34, 34, 34};
              return (get_linear_index(
                          cell_positions[a][0], cell_positions[a][1],
                          cell_positions[a][2], n_q) <
                      get_linear_index(
                          cell_positions[b][0], cell_positions[b][1],
                          cell_positions[b][2], n_q));
#endif
              });
    std::string res =
        "Particle " + std::to_string(particle_id) + ": (" +
        std::to_string(particle_position[0]) + ", " +
        std::to_string(particle_position[1]) + ", " +
        std::to_string(particle_position[2]) + ") " +
        "f_part (" +
        std::to_string(particle_force[0]) + ", " +
        std::to_string(particle_force[1]) + ", " +
        std::to_string(particle_force[2]) + ");\ninterpolation fluid:\n";
    for (int i = 0; i < delta.size(); ++i) {
      res.append("pos: " +
                 std::to_string(cell_positions[coupling_order[i]][0]) + ", " +
                 std::to_string(cell_positions[coupling_order[i]][1]) + ", " +
                 std::to_string(cell_positions[coupling_order[i]][2]) + "; " +
                 "wrapped pos: " +
                 std::to_string(wrapped_cell_positions[coupling_order[i]][0]) + ", " +
                 std::to_string(wrapped_cell_positions[coupling_order[i]][1]) + ", " +
                 std::to_string(wrapped_cell_positions[coupling_order[i]][2]) + "; " +
                 "delta: " +
                 std::to_string(delta[coupling_order[i]]) + "; " +
                 "fluid force: (" +
                 std::to_string(fluid_force[coupling_order[i]][0]) + ", " +
                 std::to_string(fluid_force[coupling_order[i]][1]) + ", " +
                 std::to_string(fluid_force[coupling_order[i]][2]) + ")\n");
    }
    res += "\n";
    return res;
  }
} coupling_helper_t;

#endif //ESPRESSO_COUPLING_HELPER_HPP
