
#ifndef REPART_HPP_INCLUDED
#define REPART_HPP_INCLUDED

#include <string>
#include <functional>
#include <vector>
#include <mpi.h>

namespace repart {

extern double force_calc_runtime;

struct RuntimeRecorder {
  RuntimeRecorder(double& t): t(t) { t = MPI_Wtime(); }
  ~RuntimeRecorder() { t = MPI_Wtime() - t; }
private:
  double& t;
};


// Print general information about cell process mapping
void print_cell_info(const std::string& prefix, const std::string& method);

/** Represents a linear combination of single metric functions.
 */
struct metric {
  using metric_func = std::function<void(std::vector<double>&)>;

  /** Constructor. Might throw a std::invalid_argument exception if desc is not understood.
   * Metric description strings are linear combinations of single metrics.
   * E.g. "2.0*ncells +1.7*nghostpart"
   * The space after the metric name ("ncell") is mandatory.
   * Factor, multiplication and addition sign are mandatory. Negative constants
   * are only allowed for the first factor. For further use subtraction instead
   * of addition, e.g. "-1.0*ncells -1.7*nghostpart".
   * Single metric names are also acceptable and interpreted as "1.0<name>".
   * Valid metrics are: ncells, npart, ndistpairs, nforcepairs, nbondedia,
   * nghostcells, nghostpart, runtime and rand.
   * \param desc string to describe the metric
   */
  metric(const std::string& desc);

  /** Calculates the metric and returns the weights.
   * \return vector of weights. Length: local_cells.n
   */
  std::vector<double> operator()() const;

private:
  void parse_metric_desc(const std::string& desc);

  std::vector<std::pair<double, metric_func>> mdesc;
};

}

#endif

