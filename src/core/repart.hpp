
#ifndef REPART_HPP_INCLUDED
#define REPART_HPP_INCLUDED

#include <string>
#include <functional>
#include <vector>
#include <mpi.h>

namespace repart {

/** Represents a linear combination of single metric functions.
 */
struct metric {
  using metric_func = std::function<void(std::vector<double>&)>;

  metric() {}
  metric(const std::string& desc) {
    set_metric(desc);
  }

  /** Metric setter.
   * Might throw a std::invalid_argument exception if desc is not understood.
   * Metric description strings are linear combinations of
   * single metrics, for example: "2.0*ncells + 1.7*nghostpart". Negative
   * constants are only allowed for the first factor. For further use
   * subtraction instead of addition, e.g. "-1.0*ncells - 1.7*nghostpart".
   * Single metric names are also acceptable and interpreted as "1.0*<name>".
   * Valid metrics are: ncells, npart, ndistpairs, nforcepairs, nbondedia,
   * nghostcells, nghostpart and rand.
   * \param desc string to describe the metric
   */
  void set_metric(const std::string& desc);

  /** Calculates the metric and returns the weights.
   * \return vector of weights. Length: local_cells.n
   */
  std::vector<double> operator()() const;

  double curload() const;
  double paverage() const;
  double pmax() const;
  double pmin() const;
  double pimbalance() const;

  std::array<double, 4> pmax_avg_min_imb() const;


private:
  void parse_metric_desc(const std::string& desc);

  std::vector<std::pair<double, metric_func>> mdesc;
};

}

#endif

