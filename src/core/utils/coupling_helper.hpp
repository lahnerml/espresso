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
      for (auto &p : cell_positions) {
        for (int i = 0; i < 3; ++i) {
          if (p[i] == 0) {
            p[i] = 31;
          } else if (p[i] == 33) {
            p[i] = 0;
          } else {
            --p[i];
          }
        }
      }
      cell_positions_wrapped = true;
    }
  }

  std::string print(std::vector<uint64_t> &coupling_order) {
    preprocess_position();
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
