
#ifndef REPART_HPP_INCLUDED
#define REPART_HPP_INCLUDED

#include <string>
#include <functional>
#include <vector>

namespace repart {


// Print general information about cell process mapping
void print_cell_info(const std::string& prefix, const std::string& method);

/** Represents a linear combination of single metric functions.
 */
struct metric {
  using metric_func = std::function<void(std::vector<double>&)>;

  /** Constructor. Might throw a std::invalid_argument exception if desc is not understood.
   * \param desc string to describe the metric, e.g. "2.0*ncells+1.7*nghostpart"
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

