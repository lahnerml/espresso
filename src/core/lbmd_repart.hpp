#ifndef LBMD_REPART_HPP_
#define LBMD_REPART_HPP_

#include <string>
#include <vector>

#if (defined(LB_ADAPTIVE) || defined(DD_P4EST))

namespace lbmd {

void repart_all(const std::vector<std::string>& metrics, const std::vector<double>& alphas);

}

#endif

#endif

