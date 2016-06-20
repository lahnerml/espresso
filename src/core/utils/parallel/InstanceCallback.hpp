/*
  Copyright (C) 2010,2011,2012,2013,2014,2015,2016 The ESPResSo project
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

#ifndef COMMUNICATION_INSTANCE_CALLBACK_HPP
#define COMMUNICATION_INSTANCE_CALLBACK_HPP

#include <functional>

#include "MpiCallbacks.hpp"
#include "utils/ParallelFactory.hpp"

namespace Communication {

/**
 * @brief Add a mpi callback to a class.
 *
 * This is a RAII class to register a mpi callbcack
 * per instance. The callback has the same livetime as
 * the instance: It is created in the constructor
 * and removed in the destructor.
 * Clients have to implement the mpi_slave function
 * which is the actual callback.
 */
class InstanceCallback {
public:
  InstanceCallback() {
    using std::placeholders::_1;
    using std::placeholders::_2;
    Communication::MpiCallbacks::function_type f;

    /* Bind member function to this instance */
    auto cb = std::bind(&InstanceCallback::mpi_slave, this, _1, _2);

    m_callback_id = Communication::mpiCallbacks().add(cb);
  }

  ~InstanceCallback() { Communication::mpiCallbacks().remove(m_callback_id); }

  /**
   * @brief Run the callback function on the slave.
   *
   * The callback is not run on the calling node,
   * this can be implemented in the derived class
   * if needed.
   */
  void call(int a, int b) {
    /* Call can only run on the master */
    assert(Communication::mpiCallbacks().is_root());

    Communication::mpiCallbacks().call(m_callback_id, a, b);
  }

private:
  /* mpi_slave should be private because it is virtual,
   * but the callback system has to call it. */
  friend Communication::MpiCallbacks;
  virtual void mpi_slave(int, int) = 0;

  /* Id of the encapsulated callback */
  int m_callback_id;
};

} /* namespace Communication */

#endif
