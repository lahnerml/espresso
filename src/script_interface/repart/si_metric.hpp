/*
  Copyright (C) 2010,2011,2012,2013,2014 The ESPResSo project
  Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
  Max-Planck-Institute for Polymer Research, Theory Group

  This file is part of ESPResSo.

  ESPResSo is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ESPResSo is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ESPRESSO_SCRIPTINTERFACE_REPART_METRIC_HPP
#define ESPRESSO_SCRIPTINTERFACE_REPART_METRIC_HPP

#include "ScriptInterface.hpp"
#include "repart.hpp"
#include <string>

namespace ScriptInterface {
namespace Repart {

class SIMetric : public ScriptInterfaceBase {
public:
  SIMetric() {}

  ParameterMap valid_parameters() const override {
    return {{"metric", {ParameterType::STRING, true}}};
  }

  VariantMap get_parameters() const override {
    return {{"metric", m_metric_desc}};
  }

  void set_parameter(const std::string &name, const Variant &value) override {
    if (name == "metric") {
      m_metric_desc = boost::get<std::string>(value);
      m_metric.set_metric(m_metric_desc);
    }
  }

  Variant call_method(const std::string &name,
                      const VariantMap &parameters) override {
    if (name == "average")
      return m_metric.paverage();
    else if (name == "maximum")
      return m_metric.pmax();
    else if (name == "imbalance")
      return m_metric.pimbalance();
    else
      return {};
  }

  const ::repart::metric &get_metric() const {
    return m_metric;
  }

private:
  std::string m_metric_desc;
  ::repart::metric m_metric;
};

}
}

#endif
