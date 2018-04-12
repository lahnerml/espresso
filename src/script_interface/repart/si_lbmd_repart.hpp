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

#ifndef ESPRESSO_SCRIPTINTERFACE_LBMD_REPART_HPP
#define ESPRESSO_SCRIPTINTERFACE_LBMD_REPART_HPP

#include "ScriptInterface.hpp"
#include "lbmd_repart.hpp"
#include <string>

namespace ScriptInterface {
namespace Repart {

class SILBMD_Repart : public ScriptInterfaceBase {
public:
  SILBMD_Repart() {}

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
      auto lbm = boost::get<std::string>(parameters.at("lb_metric"));
      auto mdm = boost::get<std::string>(parameters.at("md_metric"));
      auto lba = boost::get<double>(parameters.at("lb_alpha"));
      auto mda = boost::get<double>(parameters.at("md_alpha"));

      // Same ordering as forest_order
      std::vector<std::string> ms = {mdm, lbm};
      std::vector<double> as = {mda, lba};

#ifdef DD_P4EST
      if (cell_structure.type != CELL_STRUCTURE_P4EST) {
        std::cerr << "Trying to repartition non-p4est cell system." << std::endl
                  << "Continuing without repart." << std::endl;
      }
#endif

#if defined(DD_P4EST) || defined(LB_ADAPTIVE)
      lbmd::repart_all(ms, as);
#endif
    }
    return {};
  }
};

}
}

#endif
