/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/sycl/seeding/seed_finding.hpp"
#include "traccc/sycl/seeding/spacepoint_binning.hpp"
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

/// Main algorithm for performing the track seeding using oneAPI/SYCL
class seeding_algorithm : public algorithm<vecmem::data::vector_buffer<seed>(
                              const spacepoint_container_types::const_view&)> {

    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param mr The memory resource to use
    /// @param queue The SYCL queue to work with
    ///
    seeding_algorithm(vecmem::memory_resource& mr, vecmem::memory_resource& device_mr, queue_wrapper queue);

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints_view A view of all spacepoints in the event
    /// @return The track seeds reconstructed from the spacepoints
    ///
    output_type operator()(const spacepoint_container_types::const_view&
                               spacepoints) const override;

    private:
    /// Sub-algorithm performing the spacepoint binning
    spacepoint_binning m_spacepoint_binning;
    /// Sub-algorithm performing the seed finding
    seed_finding m_seed_finding;

};  // class seeding_algorithm

}  // namespace traccc::sycl
