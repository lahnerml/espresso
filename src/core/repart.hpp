
#ifndef REPART_HPP_INCLUDED
#define REPART_HPP_INCLUDED

#include <string>
#include <functional>
#include <vector>

namespace repart {

// Print general information about cell process mapping
void print_cell_info(const std::string& prefix, const std::string& method);

// Get the appropriate metric function described in string "desc".
// Throws a std::invalid_argument exception if no such metric is available
std::function<void(std::vector<double>&)>
get_metric_func(const std::string& desc);

}

#endif

