/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/sycl/seeding/seeding_algorithm.hpp"

// Project include(s).
#include "traccc/seeding/detail/seeding_config.hpp"

// System include(s).
#include <cmath>

namespace traccc::sycl {

seeding_algorithm::seeding_algorithm(const traccc::memory_resource& mr,
                                     const queue_wrapper& queue)
    : m_spacepoint_binning(default_seedfinder_config(),
                           default_spacepoint_grid_config(), mr, queue),
      m_seed_finding(default_seedfinder_config(), mr, queue) {}

vecmem::data::vector_buffer<seed> seeding_algorithm::operator()(
    const spacepoint_container_types::const_view& spacepoints_view) const {

    return m_seed_finding(spacepoints_view,
                          m_spacepoint_binning(spacepoints_view));
}

vecmem::data::vector_buffer<seed> seeding_algorithm::operator()(
    const spacepoint_container_types::buffer& spacepoints_buffer) const {

    return m_seed_finding(spacepoints_buffer,
                          m_spacepoint_binning(spacepoints_buffer));
}

}  // namespace traccc::sycl
