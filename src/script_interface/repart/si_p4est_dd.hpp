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

#ifndef ESPRESSO_SCRIPTINTERFACE_P4EST_DD_HPP
#define ESPRESSO_SCRIPTINTERFACE_P4EST_DD_HPP

#include "ScriptInterface.hpp"
#include "p4est_dd.hpp"
#include "repart.hpp"
#include <string>

extern int this_node;

namespace ScriptInterface {
namespace Repart {

class SIP4estDD : public ScriptInterfaceBase {
public:
  SIP4estDD() {}

  ParameterMap valid_parameters() const override {
    return {};
  }

  VariantMap get_parameters() const override {
    return {};
  }

  void set_parameter(const std::string &name, const Variant &value) override {
    (void) name;
    (void) value;
  }

  Variant call_method(const std::string &name,
                      const VariantMap &parameters) override {
    if (name == "repart") {
      std::string m = boost::get<std::string>(parameters.at("metric"));
      bool verbose = boost::get<bool>(parameters.at("verbose"));

      if (verbose && this_node == 0)
        std::cout << "SIP4estDD::call_method(repart) with metric: " << m << std::endl;

      if (cell_structure.type != CELL_STRUCTURE_P4EST && this_node == 0) {
        std::cerr << "Trying to repartition non-p4est cell system." << std::endl
                  << "Continuing without repart." << std::endl;
      }
      p4est_dd_repartition(m, verbose);
    } else if (name == "is_lb_adaptive_defined") {
#ifdef LB_ADAPTIVE
      return true;
#else
      return false;
#endif
    }
    return {};
  }
};

}
}

#endif
