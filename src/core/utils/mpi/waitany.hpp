#ifndef WAITANY_HPP_INCLUDED
#define WAITANY_HPP_INCLUDED

#include <utility> // std::pair
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>


namespace Utils {
namespace Mpi {
// Manual implementation of boost::mpi::wait_any until fixed
// version is in the releases and adopted by distributions.
// Returns an empty status and "last" if all requests are null
// (corresponds to MPI_Waitany returning MPI_UNDEFINED).
template <typename It>
std::pair<boost::mpi::status, It>
wait_any(It first, It last) {
  bool exists_non_null;
  boost::optional<boost::mpi::status> os;

  while (true) {
    exists_non_null = false;
    for (It el = first; el != last; el++) {
      // Element done. Do not call test() on it
      if (el->m_requests[0] == MPI_REQUEST_NULL &&
          el->m_requests[1] == MPI_REQUEST_NULL)
        continue;

      exists_non_null = true;
      if ((os = el->test()))
        return std::make_pair(*os, el);
    }

    // Prevent infinite loop
    if (!exists_non_null)
      return std::make_pair(boost::mpi::status{}, last);
  }
}
}
}

#endif
