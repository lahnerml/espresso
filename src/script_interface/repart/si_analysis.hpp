#ifndef ESPRESSO_SCRIPTINTERFACE_REPART_ANALYSIS_HPP
#define ESPRESSO_SCRIPTINTERFACE_REPART_ANALYSIS_HPP

#include "ScriptInterface.hpp"
#include "repart.hpp"
#include <string>

extern int n_nodes;
#include "communication.hpp"

namespace ScriptInterface {
namespace Repart {

static std::vector<double> gather(double val)
{
  std::vector<double> v(n_nodes);
  MPI_Gather(&val, 1, MPI_DOUBLE, v.data(), 1, MPI_DOUBLE, 0, comm_cart);
  return v;
}

static double* variable(const std::string name)
{
  using namespace repart;
  if (name == "ivv")
    return &ivv_runtime;
  else if (name == "fc")
    return &fc_runtime;
  else if (name == "lc")
    return &lc_runtime;
  else
    return nullptr;
}

class SIAnalysis : public ScriptInterfaceBase {
public:
  SIAnalysis() {}

  Variant call_method(const std::string &name,
                      const VariantMap &parameters) override {
    using namespace repart;
    if (name == "runtime") {
      double *var = variable(boost::get<std::string>(parameters.at("funct")));
      if (var)
        return gather(*var);
    } else if (name == "reset") {
      auto m = boost::get<std::string>(parameters.at("funct"));
      if (m == "all")
        fc_runtime = lc_runtime = ivv_runtime = 0;
      else
        *variable(m) = 0.0;
    }

    return {};
  }
};

}
}

#endif

