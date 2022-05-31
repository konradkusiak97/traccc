/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/get_prefix_sum.hpp"
#include "traccc/edm/cell.hpp"

namespace traccc::device {

/// Function that finds the clusters using sparse_ccl algorithm
///
/// It saves the cluster indices for each module in a jagged vector
/// and it counts how many clusters in total were found
///
/// @param parameter-name description
/// @param parameter-name description
/// @param parameter-name description
/// @param parameter-name description
///
TRACCC_HOST_DEVICE
void find_clusters(
    const cell_container_types::const_view& cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    unsigned int& total_clusters,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view);

}   // namespace traccc::device