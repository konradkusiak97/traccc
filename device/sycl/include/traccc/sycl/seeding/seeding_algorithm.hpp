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

// traccc library include(s).
#include "traccc/utils/memory_resource.hpp"

namespace {

/// Helper function that would produce a default seed-finder configuration
traccc::seedfinder_config default_seedfinder_config() {

    traccc::seedfinder_config config;
    config.highland = 13.6 * std::sqrt(config.radLengthPerSeed) *
                      (1 + 0.038 * std::log(config.radLengthPerSeed));
    float maxScatteringAngle = config.highland / config.minPt;
    config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
    // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV
    // and millimeter
    // TODO: change using ACTS units
    config.pTPerHelixRadius = 300. * config.bFieldInZ;
    config.minHelixDiameter2 =
        std::pow(config.minPt * 2 / config.pTPerHelixRadius, 2);
    config.pT2perRadius =
        std::pow(config.highland / config.pTPerHelixRadius, 2);
    return config;
}

/// Helper function that would produce a default spacepoint grid configuration
traccc::spacepoint_grid_config default_spacepoint_grid_config() {

    traccc::seedfinder_config config = default_seedfinder_config();
    traccc::spacepoint_grid_config grid_config;
    grid_config.bFieldInZ = config.bFieldInZ;
    grid_config.minPt = config.minPt;
    grid_config.rMax = config.rMax;
    grid_config.zMax = config.zMax;
    grid_config.zMin = config.zMin;
    grid_config.deltaRMax = config.deltaRMax;
    grid_config.cotThetaMax = config.cotThetaMax;
    return grid_config;
}

}  // namespace
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
namespace traccc::sycl {

/// Main algorithm for performing the track seeding using oneAPI/SYCL
class seeding_algorithm : public algorithm<vecmem::data::vector_buffer<seed>(
                              const spacepoint_container_types::const_view&)>,
                          public algorithm<vecmem::data::vector_buffer<seed>(
                              const spacepoint_container_types::buffer&)> {

    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param mr is a struct of memory resources (shared or host & device)
    /// @param queue The SYCL queue to work with
    ///
    seeding_algorithm(const traccc::memory_resource& mr,
                      const queue_wrapper& queue);

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints_view is a view of all spacepoints in the event
    /// @return the track seeds reconstructed from the spacepoints
    ///
    vecmem::data::vector_buffer<seed> operator()(
        const spacepoint_container_types::const_view& spacepoints_view)
        const override;

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints_buffer A buffer with all spacepoints in the event
    /// @return The track seeds reconstructed from the spacepoints
    ///
    vecmem::data::vector_buffer<seed> operator()(
        const spacepoint_container_types::buffer& spacepoints_buffer)
        const override;

    private:
    /// Sub-algorithm performing the spacepoint binning
    spacepoint_binning m_spacepoint_binning;
    /// Sub-algorithm performing the seed finding
    seed_finding m_seed_finding;

};  // class seeding_algorithm

}  // namespace traccc::sycl
